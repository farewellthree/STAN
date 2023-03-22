import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPConfig
from timm.models.layers import trunc_normal_

from mmaction.core.hooks.fp16_utils import auto_fp16

from einops import rearrange
from ..builder import BACKBONES
from .clip_adapter import *


@BACKBONES.register_module()
class BERTCLIPPretrained(nn.Module):
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
    
    @auto_fp16()
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


