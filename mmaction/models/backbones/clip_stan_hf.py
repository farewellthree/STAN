from webbrowser import get
import torch
import copy
import torch.nn as nn
import numpy as np
from mmaction.registry import MODELS

from einops import rearrange

from transformers import CLIPModel, CLIPConfig
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPMLP
from mmcv.cnn.bricks import DropPath
from mmengine.model import constant_init
from torch.utils import checkpoint

from mmaction.utils.mask_generator import TubeMaskingGenerator, RandomMaskingGenerator

def inflate_weight(weight_2d, time_dim, center=True):
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d 

@MODELS.register_module()
class VITCLIPPretrained(nn.Module):
    def __init__(self, pretrained_model="openai/clip-vit-base-patch32", clip_weight=None, return_mean=True, 
                 patch_3d=False, return_all=False, **kwargs):
        super().__init__()
        print ("==== CLIP Vision ====")
        
        if clip_weight:
            configuration = CLIPConfig().from_pretrained(clip_weight)
            clip_model = CLIPModel.from_pretrained(clip_weight, config=configuration)
        else:
            configuration = CLIPConfig().from_pretrained(pretrained_model)
            clip_model = CLIPModel.from_pretrained(pretrained_model, config=configuration)
        
        self.config = configuration
        self.patch_3d = patch_3d
        self.return_all = return_all
    
        self.num_patches = (configuration.vision_config.image_size // configuration.vision_config.patch_size) ** 2
        self.embed_dim = configuration.vision_config.hidden_size
        self.projection_dim = configuration.vision_config.projection_dim
        self.patch_size = configuration.vision_config.patch_size
        
        # clip 0.0138,  0.2357, -0.1285,  ...,  0.0171, -0.3332, -0.2366
        self.class_embedding = clip_model.vision_model.embeddings.class_embedding
        self.patch_embedding = clip_model.vision_model.embeddings.patch_embedding
        self.position_embedding = clip_model.vision_model.embeddings.position_embedding

        self.pre_layrnorm = clip_model.vision_model.pre_layrnorm
        self.post_layernorm = clip_model.vision_model.post_layernorm
        self.layers = clip_model.vision_model.encoder.layers
        self.return_mean = return_mean

        self.visual_projection = clip_model.visual_projection
        del clip_model
        
    def init_weights(self):
        pass

    def forward_embedding(self, x):
        batch_size = x.shape[0]
        if self.patch_3d:
            if not hasattr(self,'patch_embedding_3d'):
                self.patch_embedding_3d = nn.Conv3d(3, self.embed_dim, (1, self.patch_size, self.patch_size), 
                    (1, self.patch_size, self.patch_size), (0, 0, 0), bias=False)
                state_dict_2d = self.patch_embedding.state_dict()
                state_dict_3d = self.patch_embedding_3d.state_dict()
                for k,v in state_dict_2d.items():
                    state_dict_3d[k] = inflate_weight(v,1)
                self.patch_embedding_3d.load_state_dict(state_dict_3d)
                self.patch_embedding_3d.to(x.device)
                del self.patch_embedding
                
            x = rearrange(x,'(b t) d w h-> b d t w h', t=self.T)
            patch_embeds = self.patch_embedding_3d(x)
            patch_embeds = rearrange(patch_embeds,'b d t w h-> (b t) (w h) d')
        else:
            patch_embeds = self.patch_embedding(x)
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_ids = torch.arange(self.num_patches + 1, dtype=torch.long, device=x.device).expand((1, -1))
        embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings

    def forward_patch(self, x, attention_mask=None):
        x = self.pre_layrnorm(x)
        encoder_states = ()
        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (x,)
            layer_outputs = encoder_layer(x, attention_mask = attention_mask, causal_attention_mask=None)
            x = layer_outputs[0]
        encoder_states = encoder_states + (x,)
        cls_token = x[:, 0, :]
        cls_token = self.post_layernorm(cls_token)

        return encoder_states, cls_token
    
    def result_process(self, vision_outputs):
        x_tokens = vision_outputs[0]
        x_cls = vision_outputs[1]

        x_cls = self.visual_projection(x_cls)

        x_cls = rearrange(x_cls, '(b t) d -> b t d', t=self.T)
        #x_tokens = rearrange(x_tokens, '(b t) l d -> b (t l) d', t=self.T)

        if self.return_mean:
            x_cls = x_cls.mean(1)
        return x_tokens, x_cls

    def forward(self, x, return_all=False, **kwargs):
        if x.ndim == 5:
            # B, 3, num_frames, 224, 224
            if x.shape[1]==3:
                B, D, T, H, W = x.shape             
                x = x.permute(0, 2, 1, 3, 4)
            else:
                B, T, D, H, W = x.shape   
            x = x.reshape((-1,) + x.shape[2:])
        else:
            B, _, _, _ = x.shape
            T = 1
        self.T = T
        
        embeddings = self.forward_embedding(x)
        vision_outputs = self.forward_patch(embeddings)
        x_tokens, x_cls = self.result_process(vision_outputs)
        
        if self.return_all and self.training:
            return x_tokens, x_cls
        #return x_tokens, x_cls
        return x_cls

@MODELS.register_module()
class VITCLIPPretrained_STAN(VITCLIPPretrained):
    def __init__(self, depth=4, cls_residue=False, time_module="selfattn",
        pretrained_model="openai/clip-vit-base-patch32", clip_weight=None, all_patch=False,
        gradient_checkpointing=False, **kwargs): 
        super().__init__(clip_weight=clip_weight, **kwargs)

        if clip_weight:
            configuration = CLIPConfig().from_pretrained(clip_weight)
        else:
            configuration = CLIPConfig().from_pretrained(pretrained_model)

        self.depth = depth
        self.clip_weight = clip_weight
        self.cls_residue = cls_residue
        self.time_module = time_module
        self.depth = depth
        self.gradient_checkpointing = gradient_checkpointing
        self.all_patch = all_patch

        dpr = np.linspace(0, 0.1, depth)
        self.STAN_S_layers = nn.ModuleList([CLIPLayer_Spatial(configuration.vision_config, 8, dpr[i]) for i in range(depth)])
        
        if time_module=="selfattn":
            self.STAN_T_layers = nn.ModuleList([CLIPLayer_AttnTime(configuration.vision_config, 8, dpr[i]) for i in range(depth)])
        elif time_module=="conv":
            self.STAN_T_layers = nn.ModuleList([CLIPLayer_ConvTime(self.embed_dim, 8, dpr[i]) for i in range(depth)])

        #self.STAN_pos_embed = nn.Embedding(self.num_patches + 1, self.embed_dim)
        self.STAN_time_embed = nn.Embedding(64, self.embed_dim)
        self.drop_after_pos = nn.Dropout(p=0)
        self.drop_after_time = nn.Dropout(p=0)

        self.init_weights()

    def init_weights(self):
        totle_depth = len(self.layers)
        layer_para = self.layers.state_dict()
        spatial_para = {}
        load_start = totle_depth - self.depth
        for k, v in layer_para.items():
            num_layer = int(k.split(".")[0])
            if num_layer >= load_start:
                spatial_para[k.replace(str(num_layer),str(num_layer-load_start),1)] = v.clone()
        self.STAN_S_layers.load_state_dict(spatial_para)
       
    def forward_patch(self, x):
        x = self.pre_layrnorm(x)
        totle = len(self.layers)
        x2 = None
        encoder_states = ()
        for idx, encoder_layer in enumerate(self.layers):
            if x2 is None:
                encoder_states = encoder_states + (x,)
            else:
                encoder_states = encoder_states + ((x,x2),)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = checkpoint.checkpoint(encoder_layer, x, None, None)
            else: 
                layer_outputs = encoder_layer(x, attention_mask = None, causal_attention_mask=None)
            x = layer_outputs[0]

            if idx >= totle-self.depth: 
                num_layer = idx + self.depth - totle
                x2 = self.forword_timeModule(x, x2, num_layer, self.T)

        encoder_states = encoder_states + ((x,x2),)
        
        cls_token = x[:, 0] + x2[:, 0].repeat(1, self.T).view(x2.size(0) * self.T, -1)
        cls_token = self.post_layernorm(cls_token)

        return encoder_states, cls_token

    def forword_timeModule(self, x1, x2, num_layer, T):
        x1 = rearrange(x1, '(b t) l d -> b t l d', t=T)

        self.STAN_S_layers[num_layer].t = T
        self.STAN_T_layers[num_layer].t = T

        if x2 is not None:
            cls_token_ori = x1[:, :, 0, :]
            cls_token = cls_token_ori.mean(dim=1).unsqueeze(1)
            x1 = x1[:, :, 1:, :]
            x1 = rearrange(x1, 'b t l d -> b (l t) d')
            x1 = torch.cat((cls_token, x1), dim=1)

            if not self.cls_residue:
                x = x2 + x1
            else:
                if self.training:
                    cls_token1 = cls_token_ori[:,0::2,:].mean(dim=1).unsqueeze(1)
                else:
                    cls_token1 = cls_token_ori.mean(dim=1).unsqueeze(1)
                
                x1 = torch.cat((cls_token1.repeat(1,1,1), x1[:, 1:, :]), dim=1)
                x = x2 + x1
        else:
            x = x1
        
        if num_layer==0:
            x = self.input_ini(x)

        if self.gradient_checkpointing and self.training:
            x = checkpoint.checkpoint(self.STAN_T_layers[num_layer],x)
            x = checkpoint.checkpoint(self.STAN_S_layers[num_layer],x,None,None)
        else: 
            x = self.STAN_T_layers[num_layer](x)
            x = self.STAN_S_layers[num_layer](x, None, None)
        return x 

    def input_ini(self, x):
        cls_old = x[:, :, 0, :].mean(dim=1).unsqueeze(1)
        x = x[:,:,1:,:]
        B,T,L,D = x.size()
        x = rearrange(x, 'b t l d -> (b t) l d')
        
        cls_tokens = self.class_embedding.expand(x.size(0), 1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_embed = self.position_embedding(position_ids)
        x = x + pos_embed
        x = self.drop_after_pos(x)
        cls = x[:B, 0, :].unsqueeze(1)
        x = rearrange(x[:, 1:, :], '(b t) l d -> (b l) t d', b=B)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        time_embed = self.STAN_time_embed(position_ids)
        x = x + time_embed
        x = self.drop_after_time(x)
        x = rearrange(x, '(b l) t d -> b (l t) d', b=B)
        cls = (cls_old + cls) / 2
        x = torch.cat((cls, x), dim=1)
        return x 

class CLIPLayer_Spatial(nn.Module):
    def __init__(self, config: CLIPConfig, T, layer_num=0.1):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = DropPath(
            layer_num) if layer_num > 0. else nn.Identity()
        self.t = T

       
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ):
        residual = hidden_states

        #init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
        #query_s = hidden_states[:, 1:, :]
        
        init_cls_token = hidden_states[:, :1, :]
        query_s = hidden_states[:, 1:, :]

        b, pt, m = query_s.size()
        p, t = pt // self.t, self.t
        cls_token = init_cls_token.unsqueeze(1).repeat(1, t, 1, 1).reshape(b * t, 1, m) #can I do?
        #cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t, m).unsqueeze(1)
        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        hidden_states = torch.cat((cls_token, query_s), 1)
        #hidden_states = self.process.before(hidden_states)

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
        )

        res_spatial = self.dropout_layer(
            self.proj_drop(hidden_states.contiguous()))
        cls_token = res_spatial[:, :1, :].reshape(b, self.t, 1, m)
        cls_token = torch.mean(cls_token, 1)
        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(
            res_spatial[:, 1:, :], '(b t) p m -> b (p t) m', p=p, t=self.t)
        hidden_states = torch.cat((cls_token, res_spatial), 1)
        #hidden_states = self.process.after(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class CLIPLayer_AttnTime(nn.Module):
    def __init__(self, config: CLIPConfig, T, layer_num=0.1):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)

        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = DropPath(
            layer_num) if layer_num > 0. else nn.Identity()
        #-1.0633e-04,  2.3007e-04, -6.0737e-05,  ...,  2.2769e-05,
        self.temporal_fc = nn.Linear(self.embed_dim, self.embed_dim)
        constant_init(self.temporal_fc, val=0, bias=0)
        self.t = T 

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        causal_attention_mask=None,
    ):
        residual = hidden_states[:, 1:, :]


        #init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
        #query_t = hidden_states[:, 1:, :]
        init_cls_token = hidden_states[:, :1, :]
        query_t = hidden_states[:, 1:, :]
        # query_t [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_t.size()
        p, t = pt // self.t, self.t
        hidden_states = query_t.reshape(b * p, t, m)

        #init_cls_token, hidden_states = self.process.before(hidden_states)

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
        )

        res_temporal = self.dropout_layer(
            self.proj_drop(hidden_states.contiguous()))
        res_temporal = self.temporal_fc(res_temporal)
        # res_temporal [batch_size, num_patches * num_frames, embed_dims]
        hidden_states = res_temporal.reshape(b, p * self.t, m)
        #hidden_states = self.process.after(hidden_states)


        hidden_states = residual + hidden_states
        hidden_states = torch.cat((init_cls_token, hidden_states), 1)
        outputs = hidden_states

        return outputs
    
@MODELS.register_module()
class CLIPTextPretrained(nn.Module):
    def __init__(self, pretrained_model="openai/clip-vit-base-patch32", 
        all_proj=False, clip_weight=None, hidden_dim=768 , max_pos=77, **kwargs):
        super().__init__()
        print ("==== CLIP TextModel ====")
        if clip_weight:
            configuration = CLIPConfig().from_pretrained(clip_weight)
            #configuration.text_config.max_position_embeddings = max_pos
            clip_model = CLIPModel.from_pretrained(clip_weight, config=configuration)
        else:
            configuration = CLIPConfig().from_pretrained(pretrained_model)
            #configuration.text_config.max_position_embeddings = max_pos
            clip_model = CLIPModel.from_pretrained(pretrained_model, config=configuration)

        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        
        self.with_project = False
        self.all_proj = all_proj
        del clip_model

    def init_weights(self):
        if self.with_project:
            if isinstance(self.projector, nn.Linear):
                nn.init.xavier_uniform_(self.projector.weight)
                if isinstance(self.projector, nn.Linear) and self.projector.bias is not None:
                    self.projector.bias.data.zero_()
    
    def forward(self, token_ids=None, input_mask=None, **kwargs):
        # token_ids = token_ids.int()
        # input_mask = input_mask.int()
        text_outputs = self.text_model(input_ids=token_ids, attention_mask=input_mask)
        text_tokens = text_outputs[0]
        text_cls = text_outputs[1]
        text_cls = self.text_projection(text_cls)
        if self.all_proj:
            text_tokens = self.text_projection(text_tokens)
        return text_tokens, text_cls
    