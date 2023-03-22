import torch
import time
from torch import nn
import numpy as np
import torch.nn.functional as F
from mmaction.core.hooks.fp16_utils import auto_fp16
from ..builder import RECOGNIZERS, build_backbone, build_head, build_loss
from .base import BaseRecognizer
import random
from einops import rearrange

@RECOGNIZERS.register_module()
class VisualTextFinetuneModelClip(BaseRecognizer):
    def __init__(self,
                 text_backbone=None,
                 interaction_head= None,
                 loss_type=None,
                 task=None,
                 ssl_head=None,
                 logit_scale=100,
                 cache_text=True,
                 from_scratch=False,
                 loss_weight=None,  
                 separate_test=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.text_backbone = build_backbone(text_backbone)
        self.from_scratch = from_scratch
        self.loss_weight = loss_weight
        self.separate_test = separate_test
        self.task = task
        self.interaction_head = build_head(interaction_head) if interaction_head is not None else None 

        if self.task == 'retrieval_ssl':
            self.ssl_head = build_head(ssl_head) if ssl_head is not None else None 
            self.loss_func = build_loss(loss_type)
        elif self.task == 'zero_shot_clf':
            self.loss_func = build_loss(loss_type)
            self.cache_text = cache_text
            self.logit_scale = logit_scale
            if cache_text:
                self.cache_text_features = None
        else:
            raise NotImplementedError(f"must have head to do downstream finetuning")
        # self.a = nn.Parameter(torch.ones(1))

    @auto_fp16(apply_to=('imgs'))
    def extract_visual_feat(self, imgs, mask=None):
        if len(imgs.size())==6:
            imgs = imgs.reshape((-1, ) + imgs.shape[2:]) 
        if self.from_scratch:
            imgs = imgs / 255.0
        visual_emb = self.backbone(imgs, mask)
        return visual_emb

    def extract_text_feat(self, token_ids, input_mask):
        if len(token_ids.size())==4:
            token_ids = token_ids.reshape((-1, ) + token_ids.shape[2:])
            input_mask = input_mask.reshape((-1, ) + input_mask.shape[2:])
        text_out_with_mask, text_mask_cls = self.text_backbone(token_ids, input_mask)
        return text_out_with_mask, text_mask_cls

    @auto_fp16(apply_to=('imgs'))
    def forward_train(self, imgs, label, token_ids=None, segment_ids=None, input_mask=None, ans_ids=None, ans_mask=None, **kwargs):
        """Defines the computation performed at every call when training."""
        # text reshape:  (batch_size, num_candidates, seq_length) -> (batch_size * num_candidates, seq_length)
        losses = dict()
             
        visual_token, visual_cls = self.extract_visual_feat(imgs) # b, d, T, h, w
        if self.task == 'zero_shot_clf':
            if self.cache_text:
                if self.cache_text_features is None:
                    self.eval()
                    with torch.no_grad():
                        text_out_with_mask, text_mask_cls = self.extract_text_feat(token_ids[:1], input_mask[:1])
                        self.cache_text_features = text_mask_cls
                    self.train()
                text_mask_cls = self.cache_text_features
            else:
                text_out_with_mask, text_mask_cls = self.extract_text_feat(token_ids[:1], input_mask[:1])
            
            logit_scale = self.logit_scale
            
            if self.interaction_head is not None:
                sim_mt = self.interaction_head(text_out_with_mask, visual_cls, input_mask[:1], video_patch=visual_token)
                if hasattr(self.interaction_head, 'reco2'):
                    sim_mt = logit_scale * sim_mt
                else:
                    sim_mt = logit_scale * sim_mt.T
            else:
                visual_cls = visual_cls / visual_cls.norm(dim=-1, keepdim=True)
                text_mask_cls = text_mask_cls / text_mask_cls.norm(dim=-1, keepdim=True)
                
                sim_mt = torch.matmul(visual_cls, logit_scale * text_mask_cls.transpose(0, 1))
            
            
            cls_loss = self.loss_func(sim_mt, torch.squeeze(label))
            losses['cls_loss'] = cls_loss
            return losses

        # text feature #
        text_out_with_mask, text_mask_cls = self.extract_text_feat(token_ids, input_mask)

        #  contrastive type finetuning retrieval #
        if self.task == 'retrieval_ssl':
            retrieve_logits = None
            if self.interaction_head is not None:
                parse_idx, parse_mask = None, None
                if 'parse_idx' in kwargs:
                    parse_idx = kwargs['parse_idx']
                    parse_mask = kwargs['parse_mask']
                retrieve_logits = self.interaction_head(text_out_with_mask, visual_cls, input_mask, video_patch=visual_token, parse_idx=parse_idx, parse_mask=parse_mask)
            nce_loss = self.loss_func(visual_cls, text_mask_cls, retrieve_logits)
            losses['retrieval_nce_loss'] = nce_loss
        
        return losses


    def forward_gradcam(self, imgs, token_ids=None, segment_ids=None,input_mask=None, **kwargs):
        self.half()
        imgs = imgs.half()
        #token_ids = token_ids.half()
        #input_mask = input_mask.half()
        visual_token, visual_cls = self.extract_visual_feat(imgs)
        text_out_with_mask, text_mask_cls = self.extract_text_feat(token_ids, input_mask)
        visual_cls = visual_cls / visual_cls.norm(dim=-1, keepdim=True)
        text_mask_cls = text_mask_cls / text_mask_cls.norm(dim=-1, keepdim=True)
        
        sim_mt = torch.matmul(visual_cls, text_mask_cls.transpose(0, 1))
        return sim_mt

    @auto_fp16(apply_to=('imgs'))
    def forward_test(self, imgs, token_ids=None, segment_ids=None, input_mask=None, ans_ids=None, ans_mask=None, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        N = 1
        if len(imgs.size())==6:
            B, N, D, T, W, H = imgs.size()
        visual_token, visual_cls = self.extract_visual_feat(imgs) # b, d, T, h, w

        if self.task == 'zero_shot_clf':
            if self.cache_text:
                if self.cache_text_features is None:
                    with torch.no_grad():
                        text_out_with_mask, text_mask_cls = self.extract_text_feat(token_ids[:1], input_mask[:1])
                        self.cache_text_features = text_mask_cls    
                text_mask_cls = self.cache_text_features
            else:
                text_out_with_mask, text_mask_cls = self.extract_text_feat(token_ids[:1], input_mask[:1])
            
            logit_scale = self.logit_scale
            if self.interaction_head is not None:
                sim_mt = self.interaction_head(text_out_with_mask, visual_cls, input_mask[:1], video_patch=visual_token)
                if hasattr(self.interaction_head, 'reco2'):
                    sim_mt = logit_scale * sim_mt
                else:
                    sim_mt = logit_scale * sim_mt.T
            else:   
                visual_cls = visual_cls / visual_cls.norm(dim=-1, keepdim=True)
                text_mask_cls = text_mask_cls / text_mask_cls.norm(dim=-1, keepdim=True)
                
                sim_mt = torch.matmul(visual_cls, logit_scale * text_mask_cls.transpose(0, 1))

            if N>1:
                sim_mt = sim_mt.contiguous().view(B, N, -1)
                sim_mt = F.softmax(sim_mt, dim=2).mean(dim=1)
            return sim_mt.cpu().numpy()
        # text feature #
        text_out_with_mask, text_mask_cls = self.extract_text_feat(token_ids, input_mask)
        
        # only use the multimodal transformer for  
        if self.separate_test:
            # visual_emb = self.ssl_head.forward_vision(visual_token)
            # text_emb = self.mlm_ssl_T_head(text_out_last_hidden_state[:, 0])
            #visual_emb, text_emb = self.ssl_head(visual_token, text_out_last_hidden_state, input_mask, token_ids)
            if isinstance(visual_cls, tuple):
                visual_cls, head_cls = visual_cls
                visual_cls = visual_cls + head_cls
            
            if self.interaction_head is not None:
                #input_mask = input_mask.reshape((-1, ) + input_mask.shape[2:])
                #return visual_cls.mean(1), text_out_with_mask[torch.arange(text_out_with_mask.shape[0]), input_mask.argmin(dim=-1)-1]
                if 'patch' in self.interaction_head.mode:
                    return visual_token, text_out_with_mask
                return visual_cls, text_out_with_mask
            return visual_cls, text_mask_cls

        else:
            raise NotImplementedError("not implement the finetune test method")
            
   