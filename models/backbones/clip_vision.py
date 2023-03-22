import math
import copy
from webbrowser import get
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from mmaction.core.hooks.fp16_utils import auto_fp16
from ..builder import build_head

from einops import rearrange
from ..builder import BACKBONES
from ..heads import CLIPLayer_Spatial, CLIPLayer_AttnTime, \
     CLIPLayer_Swin3DTime, CLIPLayer_ConvTime

from transformers import CLIPModel, CLIPConfig
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPMLP


@BACKBONES.register_module()
class VITCLIPPretrained(nn.Module):
    def __init__(self, pretrained_model="openai/clip-vit-base-patch32", clip_weight=None, return_mean=True, **kwargs):
        super().__init__()
        print ("==== CLIP Vision ====")
        
        if clip_weight:
            configuration = CLIPConfig().from_pretrained(clip_weight)
            clip_model = CLIPModel.from_pretrained(clip_weight, config=configuration)
        else:
            configuration = CLIPConfig().from_pretrained(pretrained_model)
            clip_model = CLIPModel.from_pretrained(pretrained_model, config=configuration)
        
        self.config = configuration
    
        self.num_patches = (configuration.vision_config.image_size // configuration.vision_config.patch_size) ** 2
        self.embed_dim = configuration.vision_config.hidden_size
        
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
        patch_embeds = self.patch_embedding(x)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_ids = torch.arange(self.num_patches + 1, dtype=torch.long, device=x.device).expand((1, -1))
        embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings

    def forward_patch(self, x, attention_mask=None):
        x = self.pre_layrnorm(x)
        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(x, attention_mask = attention_mask, causal_attention_mask=None)
            x = layer_outputs[0]

        cls_token = x[:, 0, :]
        cls_token = self.post_layernorm(cls_token)

        return x, cls_token
    
    def result_process(self, vision_outputs):
        x_tokens = vision_outputs[0]
        x_cls = vision_outputs[1]

        x_cls = self.visual_projection(x_cls)

        x_cls = rearrange(x_cls, '(b t) d -> b t d', t=self.T)
        x_tokens = rearrange(x_tokens, '(b t) l d -> b (t l) d', t=self.T)

        if self.return_mean:
            x_cls = x_cls.mean(1)
        return x_tokens, x_cls

    @auto_fp16()
    def forward(self, x, mask=None, **kwargs):
        if x.ndim == 5:
            # B, 3, num_frames, 224, 224
            B, D, T, H, W = x.shape             
            x = x.permute(0, 2, 1, 3, 4)
            x = x.reshape((-1,) + x.shape[2:])
        else:
            B, _, _, _ = x.shape
            T = 1
        self.T = T
        if mask is not None:
            mask = mask.repeat(1,1,3,3)
   
        # vision masks ensemble into clip model
        embeddings = self.forward_embedding(x)
        vision_outputs = self.forward_patch(embeddings)
        x_tokens, x_cls = self.result_process(vision_outputs)
        
        if mask is not None:
            return x_tokens, x_cls, mask
        return x_tokens, x_cls

@BACKBONES.register_module()
class VITCLIPPretrained_STAN(VITCLIPPretrained):
    def __init__(self, depth=4, cls_residue=False, time_module="selfattn", num_cls=1,
    return_mean=True, pretrained_model="openai/clip-vit-base-patch32", clip_weight=None, **kwargs): 

        super().__init__(clip_weight=clip_weight, return_mean=return_mean, **kwargs)

        if clip_weight:
            configuration = CLIPConfig().from_pretrained(clip_weight)
        else:
            configuration = CLIPConfig().from_pretrained(pretrained_model)

        self.depth = depth
        self.clip_weight = clip_weight
        self.cls_residue = cls_residue
        self.time_module = time_module
        self.depth = depth
        self.num_cls = num_cls

        dpr = np.linspace(0, 0.1, depth)
        self.timesFPN_S_layers = nn.ModuleList([CLIPLayer_Spatial(configuration.vision_config, 8, dpr[i], num_cls=num_cls) for i in range(depth)])
        
        if time_module=="selfattn":
            self.timesFPN_T_layers = nn.ModuleList([CLIPLayer_AttnTime(configuration.vision_config, 8, dpr[i], num_cls=num_cls) for i in range(depth)])
        elif time_module=="conv":
            self.timesFPN_T_layers = nn.ModuleList([CLIPLayer_ConvTime(self.embed_dim, 8, dpr[i]) for i in range(depth)])

        self.timesFPN_pos_embed = nn.Embedding(self.num_patches + 1, self.embed_dim)
        self.timesFPN_time_embed = nn.Embedding(64, self.embed_dim)
        self.timesFPN_cls_token = nn.Parameter(torch.zeros(1, self.num_cls, self.embed_dim))
        self.drop_after_pos = nn.Dropout(p=0)
        self.drop_after_time = nn.Dropout(p=0)

        self.init_weights()

    def init_weights(self):
        totle_depth = 24 if 'L14' in self.clip_weight else 12
        layer_para = self.layers.state_dict()
        spatial_para = {}
        load_start = totle_depth - self.depth
        for k, v in layer_para.items():
            num_layer = int(k.split(".")[0])
            if num_layer >= load_start:
                spatial_para[k.replace(str(num_layer),str(num_layer-load_start),1)] = v.clone()
        self.timesFPN_S_layers.load_state_dict(spatial_para)
       
    def forward_patch(self, x, attention_mask=None):
        x = self.pre_layrnorm(x)
        totle = len(self.layers)
        x2 = None
        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(x, attention_mask = attention_mask, causal_attention_mask=None)
            x = layer_outputs[0]

            if idx >= totle-self.depth:
                num_layer = idx + self.depth - totle
                x2 = self.forword_timeModule(x, x2, num_layer, self.T)

        cls_token = x[:, 0] + x2[:, 0].repeat(1, self.T).view(x2.size(0) * self.T, -1)
        cls_token = self.post_layernorm(cls_token)

        return x, cls_token

    def forword_timeModule(self, x1, x2, num_layer, T):
        x1 = rearrange(x1, '(b t) l d -> b t l d', t=T)

        self.timesFPN_S_layers[num_layer].t = T
        self.timesFPN_T_layers[num_layer].t = T

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
                
                x1 = torch.cat((cls_token1.repeat(1,self.num_cls,1), x1[:, 1:, :]), dim=1)
                x = x2 + x1
        else:
            x = x1
        
        if num_layer==0:
            x = self.input_ini(x)
            
        x = self.timesFPN_T_layers[num_layer](x)
        x = self.timesFPN_S_layers[num_layer](x, None, None)[0]
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
        time_embed = self.timesFPN_time_embed(position_ids)
        x = x + time_embed
        x = self.drop_after_time(x)
        x = rearrange(x, '(b l) t d -> b (l t) d', b=B)
        cls = (cls_old + cls) / 2
        x = torch.cat((cls, x), dim=1)
        return x 


    