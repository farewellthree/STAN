import torch
import torch.nn as nn

from mmaction.registry import MODELS
from mmaction.models.recognizers.clip_similarity import GatherLayer


def norm(a, eps=1e-6):
    a_n = a.norm(dim=-1, keepdim=True)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm

@MODELS.register_module()
class Mug_head(nn.Module):
    def __init__(
        self,
        mode = 'mug_frame_token_simple',  
        input_dim = 512, 
        tau = 100,
        retrieval = True,
    ):
        super().__init__()
        self.mode = mode
        if mode=='drl_wti':
            self.text_weight_fc = nn.Linear(input_dim, 1)
            self.video_weight_fc = nn.Linear(input_dim, 1)
        
        self.tau = tau
        self.retrieval = retrieval
        self.fp16_enabled = False
    
    def forward(self, text_feat, video_feat, text_mask, parse_idx=None, parse_mask=None):
        _, T, D = video_feat.size()
        if len(text_mask.size())>=3:
            text_mask = text_mask.reshape((-1, ) + text_mask.shape[2:])
        if self.training and torch.cuda.is_available() and self.retrieval:  # batch merge here
            text_feat = torch.cat(GatherLayer.apply(text_feat), dim=0)
            video_feat = torch.cat(GatherLayer.apply(video_feat), dim=0)
            text_mask = torch.cat(GatherLayer.apply(text_mask), dim=0)
            if parse_mask is not None:
                parse_mask = torch.cat(GatherLayer.apply(parse_mask), dim=0)
        B = video_feat.size(0)

        if self.mode == 'drl_wti':
            text_weight = self.text_weight_fc(text_feat).squeeze(2)  # B x N_t x D -> B x N_t
            text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool), float("-inf"))
            text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

            video_weight = self.video_weight_fc(video_feat).squeeze(2) # B x N_v x D -> B x N_v
            video_weight = torch.softmax(video_weight, dim=-1)
        
        if parse_mask is not None:
            text_feat = torch.einsum('atd,at->atd', [text_feat, parse_mask])

        text_feat = norm(text_feat)
        video_feat = norm(video_feat)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])

        if self.mode == 'mug_frame_token':  
            fill = torch.HalfTensor([float("-inf")]).type_as(retrieve_logits)

            video_pre_softmax1 = torch.softmax(retrieve_logits*self.tau, dim=-1) 
            text_feat1 = torch.einsum('abtv,bvd->abtd', [video_pre_softmax1, video_feat])
            t2v_logits = torch.einsum('abtd,atd->abt', [text_feat1, text_feat])
            
            text_pre_softmax1 = torch.softmax(torch.where(retrieve_logits==0, fill, retrieve_logits)*self.tau, dim=-2)  
            video_feat1 = torch.einsum('abtv,atd->abvd', [text_pre_softmax1, text_feat])
            v2t_logits = torch.einsum('abvd,bvd->abv', [video_feat1, video_feat])
            
            text_pre_softmax2 = torch.softmax(torch.where(t2v_logits==0, fill, t2v_logits)*self.tau, dim=-1)  
            video_feat2 = torch.einsum('abt,atd->abd', [text_pre_softmax2, text_feat])

            video_pre_softmax2 = torch.softmax(v2t_logits*self.tau, dim=-1) 
            text_feat2 = torch.einsum('abv,bvd->abd', [video_pre_softmax2, video_feat])

            retrieve_logits = torch.einsum('abd,abd->ab', [video_feat2, text_feat2])

        elif self.mode == 'mug_frame_token_simple':  # more efficient at the cost of a little performance
            fill = torch.HalfTensor([float("-inf")]).type_as(retrieve_logits)

            video_pre_softmax1 = torch.softmax(retrieve_logits*self.tau, dim=-1) 
            t2v_logits = torch.sum(retrieve_logits * video_pre_softmax1, dim=-1)
            text_pre_softmax1 = torch.softmax(torch.where(retrieve_logits==0, fill, retrieve_logits)*self.tau, dim=-2)  
            v2t_logits = torch.sum(retrieve_logits * text_pre_softmax1, dim=-2)

            text_pre_softmax2 = torch.softmax(torch.where(t2v_logits==0, fill, t2v_logits)*self.tau, dim=-1)  
            t2v_logits = torch.sum(t2v_logits * text_pre_softmax2, dim=2)
            video_pre_softmax2 = torch.softmax(v2t_logits*self.tau, dim=-1) 
            v2t_logits = torch.sum(v2t_logits * video_pre_softmax2, dim=2)

            retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        elif self.mode == 'mug_frame_text':  
            x,y,t,v = retrieve_logits.size()
            textlast_index = text_mask.argmin(dim=-1)-1
            textlast_index = textlast_index.unsqueeze(-1).repeat(1,y).view(-1)

            pre_softmax1 = torch.softmax(retrieve_logits*self.tau, dim=-1)
            t2v_logits = torch.sum(retrieve_logits * pre_softmax1, dim=3)  # abtv -> abt
            v2t_logits = retrieve_logits.contiguous().view(x*y,t,v)[torch.arange(x*y), textlast_index].contiguous().view(x,y,-1)  # abtv -> abv

            t2v_logits = t2v_logits.contiguous().view(x*y,-1)[torch.arange(x*y), textlast_index].contiguous().view(x,y)
            pre_softmax2 = torch.softmax(v2t_logits*self.tau, dim=-1) 
            v2t_logits = torch.sum(v2t_logits * pre_softmax2, dim=2)
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        elif self.mode == 'drl_wti':  
            t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
            t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

            v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
            v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        
        return retrieve_logits
