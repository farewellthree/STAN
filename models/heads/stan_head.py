from mimetypes import init
from pyparsing import Opt
import torch
import math
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from functools import partial
from collections import OrderedDict
from einops import rearrange
from typing import Optional, Tuple

from mmcv.cnn.utils.weight_init import trunc_normal_

from .base import BaseHead
from mmcv.cnn import normal_init

from ..builder import HEADS
from ..builder import build_backbone

from transformers import CLIPModel, CLIPConfig
from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPAttention, CLIPMLP, CLIPEncoderLayer

from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn import constant_init
from mmaction.models.common.transformer import DividedTemporalAttentionWithNorm


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CLIPLayer_Spatial(nn.Module):
    def __init__(self, config: CLIPConfig, T, layer_num=0.1, num_cls=1):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        dropout_layer=dict(type='DropPath', drop_prob=layer_num)
        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.t = T

        self.num_cls = num_cls
        #self.process = Spatial_Process(T=T, dropout_layer=dict(type='DropPath', drop_prob=layer_num))
       
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ):
        residual = hidden_states

        #init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
        #query_s = hidden_states[:, 1:, :]
        
        init_cls_token = hidden_states[:, :self.num_cls, :]
        query_s = hidden_states[:, self.num_cls:, :]

        b, pt, m = query_s.size()
        p, t = pt // self.t, self.t
        cls_token = init_cls_token.unsqueeze(1).repeat(1, t, 1, 1).reshape(b * t, self.num_cls, m) #can I do?
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
        cls_token = res_spatial[:, :self.num_cls, :].reshape(b, self.t, self.num_cls, m)
        cls_token = torch.mean(cls_token, 1)
        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(
            res_spatial[:, self.num_cls:, :], '(b t) p m -> b (p t) m', p=p, t=self.t)
        hidden_states = torch.cat((cls_token, res_spatial), 1)
        #hidden_states = self.process.after(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs

class CLIPLayer_AttnTime(nn.Module):
    def __init__(self, config: CLIPConfig, T, layer_num=0.1, num_cls=1):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.num_cls = num_cls

        #self.ini_weight()

        dropout_layer = dict(type='DropPath', drop_prob=layer_num)
        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = build_dropout(dropout_layer) 
        self.temporal_fc = nn.Linear(self.embed_dim, self.embed_dim)
        constant_init(self.temporal_fc, val=0, bias=0)
        self.t = T 
    
    def ini_weight(self):
        xavier_uniform_(self.self_attn.k_proj.weight)
        xavier_uniform_(self.self_attn.v_proj.weight)
        xavier_uniform_(self.self_attn.q_proj.weight)

        constant_(self.self_attn.k_proj.bias, 0.)
        constant_(self.self_attn.v_proj.bias, 0.)
        constant_(self.self_attn.q_proj.bias, 0.)
        constant_(self.self_attn.out_proj.bias, 0.)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        causal_attention_mask=None,
    ):
        residual = hidden_states[:, self.num_cls:, :]


        #init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
        #query_t = hidden_states[:, 1:, :]
        init_cls_token = hidden_states[:, :self.num_cls, :]
        query_t = hidden_states[:, self.num_cls:, :]
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

class CLIPLayer_Swin3DTime(nn.Module):
    def __init__(self, swin3D_config, T=8, layer_num=0.1):
        super().__init__()
        swin3D = build_backbone(swin3D_config)
        swin3D.init_weights()
        self.layers = copy.deepcopy(swin3D.layers[-1])
        #del self.layers.blocks[-1]
        self.t = T

        dropout_layer = dict(type='DropPath', drop_prob=layer_num)
        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = build_dropout(dropout_layer) 
        self.temporal_fc = nn.Linear(swin3D.num_features, swin3D.num_features)
        constant_init(self.temporal_fc, val=0, bias=0)
        del swin3D
    
    def forward(self, hidden_states, mask=None, attention_mask=None):
        residual = hidden_states[:, 1:, :]
        init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
        hidden_states = hidden_states[:, 1:, :]

        B, TL ,D = hidden_states.size()
        T, L = self.t, TL // self.t
        WH = int(math.sqrt(L))

        hidden_states = rearrange(hidden_states, 'b (l t) d -> b t l d', t=self.t)
        hidden_states = rearrange(hidden_states, 'b t (w h) d -> b d t w h', w=WH)

        hidden_states = self.layers(hidden_states)

        hidden_states = rearrange(hidden_states, 'b d t w h -> b t (w h) d')
        hidden_states = rearrange(hidden_states, 'b t l d -> b (l t) d', t=self.t)

        hidden_states = self.dropout_layer(
            self.proj_drop(hidden_states.contiguous()))
        hidden_states = self.temporal_fc(hidden_states)

        hidden_states = residual + hidden_states

        hidden_states = torch.cat((init_cls_token, hidden_states), 1)

        outputs = hidden_states

        return outputs

class STConvpass(nn.Module):
    def __init__(self, input_dim):
        super().__init__()


        self.embed_dim = input_dim
        
        self.hidden_size = self.embed_dim
        self.adapter_downsample = nn.Conv3d(self.hidden_size, self.hidden_size, kernel_size=1)
      
        self.adapter_DWconv = nn.Conv3d(self.hidden_size, self.hidden_size, 3, 1, 1)
        
        self.adapter_activate = QuickGELU()
        self.adapter_upsample = nn.Conv3d(self.hidden_size, self.hidden_size, kernel_size=1)
        self.zero_pad = True

    def forward(self, x):
        x = self.adapter_downsample(x)
        #x = self.adapter_activate(x)

        x = self.adapter_DWconv(x)
        #x = self.adapter_activate(x)

        x = self.adapter_downsample(x)
        #x = self.adapter_activate(x)
        return x

class CLIPLayer_ConvTime(nn.Module):
    def __init__(self, hidden_size, T=8, layer_num=0.1, mode="origin"):
        super().__init__()

        if mode == "origin":
            self.layers = nn.Conv3d(hidden_size, hidden_size, 3, 1, 1)
        elif mode == "activate":
            self.layers = nn.Sequential(
                    nn.Conv3d(hidden_size, hidden_size, 3, 1, 1),
                    QuickGELU()
                )
        elif mode == "3layerST":
            self.layers = STConvpass(hidden_size)
        elif mode == "3D11":
            self.layers = nn.Conv3d(hidden_size, hidden_size, 1)
        
        self.t = T

        dropout_layer = dict(type='DropPath', drop_prob=layer_num)
        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = build_dropout(dropout_layer) 
        self.temporal_fc = nn.Linear(hidden_size, hidden_size)
        constant_init(self.temporal_fc, val=0, bias=0)
    
    def forward(self, hidden_states, mask=None, attention_mask=None):
        residual = hidden_states[:, 1:, :]
        init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
        hidden_states = hidden_states[:, 1:, :]

        B, TL ,D = hidden_states.size()
        T, L = self.t, TL // self.t
        WH = int(math.sqrt(L))

        hidden_states = rearrange(hidden_states, 'b (l t) d -> b t l d', t=self.t)
        hidden_states = rearrange(hidden_states, 'b t (w h) d -> b d t w h', w=WH)

        hidden_states = self.layers(hidden_states)

        hidden_states = rearrange(hidden_states, 'b d t w h -> b t (w h) d')
        hidden_states = rearrange(hidden_states, 'b t l d -> b (l t) d', t=self.t)

        hidden_states = self.dropout_layer(
            self.proj_drop(hidden_states.contiguous()))
        hidden_states = self.temporal_fc(hidden_states)

        hidden_states = residual + hidden_states

        hidden_states = torch.cat((init_cls_token, hidden_states), 1)

        outputs = hidden_states

        return outputs
