# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from mmengine.dist import all_gather, get_rank
from mmengine.model import BaseModel
from mmengine.structures import InstanceData, BaseDataElement
from mmengine.runner import autocast

from mmaction.registry import MODELS
from mmaction.utils import ForwardResults, OptSampleList
from mmaction.models.losses import sim_matrix
from mmaction.datasets.transforms.text_transforms import tokenize



class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> Tuple[List]:
        ctx.save_for_backward(input)
        output = all_gather(input)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank()]
        return grad_out


@MODELS.register_module()
class CLIPSimilarity_split(BaseModel):
    def __init__(
        self,
        data_preprocessor: Dict[str, Dict],
        adapter: Dict = None,
        visual_encoder = None,
        text_encoder = None,
        to_float32: bool = False,
        class_path = None,
        frozen_layers = False,
        task = "retrieval",
        tau = 0.01,
        loss: Dict = dict(type='NormSoftmaxLoss'),
    ) -> None:
        super(CLIPSimilarity_split,
              self).__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(visual_encoder)
        self.text_backbone = MODELS.build(text_encoder)
        self.loss = MODELS.build(loss)
        self.adapter = MODELS.build(adapter) if adapter is not None else None
        self.task = task
        if self.task == "recognition":
            self.cache_text = True
            self.cache_text_features = None
            with open(class_path,'r') as f:
                classes = f.readlines()
                classes = [c.strip() for c in classes]
                self.text = tokenize(classes)[:,:32]
        self.frozen_layers = frozen_layers
        self.tau = tau
        self._freeze_stages()
        #if self.frozen_layers:
        #    self.text_backbone = self.text_backbone.half()
    
    def init_weights(self):
        pass
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video."""
        frames_features = self.backbone(video)
        return frames_features

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text."""
        if self.frozen_layers:
            with torch.no_grad():
                result = self.text_backbone(text)
        else:
            result = self.text_backbone(text)
        return result

    def extract_feat(self,
                     inputs: Dict[str, torch.Tensor]) -> Tuple:
        """Extract features."""
        text_inputs = inputs['text'] if 'text' in inputs else None
        video_inputs = inputs['imgs']
        if text_inputs is None:
            text_features = None
        elif self.task=="recognition" and self.cache_text:
            text_inputs = self.text
            text_inputs = text_inputs.to(video_inputs.device)
            if self.cache_text_features is None:
                self.eval()
                with torch.no_grad():
                    text_features = self.encode_text(text_inputs)
                self.cache_text_features = text_features
                #self.cache_text_features.requires_grad = False
            text_features = self.cache_text_features
        else:
            text_features = self.encode_text(text_inputs)
        video_features = self.encode_video(video_inputs)

        return video_features, text_features

    #@autocast()
    def forward(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """Forward function."""
        if mode == 'tensor':
            return self.extract_feat(inputs)

        elif mode == 'loss':
            losses = dict()
            video_features, text_features = self.extract_feat(inputs)
            if isinstance(text_features, tuple):
                token_features, text_features = text_features
            if isinstance(video_features, tuple):
                video_tokens, video_features = video_features

            if isinstance(self.tau,float):
                logit_scale = 1 / self.tau
            else:
                logit_scale = self.tau.exp()
            
            if self.task=='retrieval':
                logit = None
                if self.adapter is not None:
                    mask = torch.where(inputs['text']==0,0,1)
                    logit = self.adapter(token_features, video_features, mask)
                else:
                    video_features = torch.cat(
                        GatherLayer.apply(video_features), dim=0)
                    text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

                sim_loss = self.loss(video_features, text_features, sim_mat=logit, scale = logit_scale)
                losses['NCE_loss'] = sim_loss

            elif self.task=='recognition':
                logits_per_video = logit_scale * sim_matrix(video_features, text_features)
                labels = [x.gt_labels.item for x in data_samples]
                labels = torch.stack(labels).to(logits_per_video.device).squeeze()
                loss = self.loss(logits_per_video,labels)
                losses['loss'] = loss
                
            return losses

        elif mode == 'predict':
            video_features, text_features = self.extract_feat(inputs)
            if isinstance(text_features, tuple):
                token_features, text_features = text_features
            if isinstance(video_features, tuple):
                video_tokens, video_features = video_features
            if self.adapter is not None:
                token_id = inputs['text'][0 ] if self.task=="recognition" else inputs['text']
                masks = torch.where(token_id==0,0,1)
                for ds, vf, tf, mask in zip(data_samples, video_features, token_features, masks):
                    tf = token_features if self.task=="recognition" else tf
                    mask = masks if self.task=="recognition" else mask
                    features = BaseDataElement(video_feature=vf, text_feature=tf, mask=mask)
                    ds.features = features
            else:
                for ds, vf, tf in zip(data_samples, video_features, text_features):
                    tf = text_features if self.task=="recognition" else tf
                    features = BaseDataElement(video_feature=vf, text_feature=tf)
                    ds.features = features
            
            return data_samples

        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode') 

    def train(self, mode: bool = True) -> None:
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_layers``."""

        if self.frozen_layers:
            for name, param in self.named_parameters():
                if "STAN" in name:
                    continue
                elif 'text_backbone' in name:
                    param.requires_grad = False
                elif 'extra_proj' in name:
                    continue
                elif 'balance' in name:
                    continue
                elif 'backbone.layers' in name:
                    layer_n = int(name.split('.')[2])
                    param.requires_grad = False
                    if layer_n>=20:
                        continue
                    else:
                        continue
                else:
                    param.requires_grad = False

    