import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..builder import LOSSES
from mmcv.runner import get_dist_info
from mmaction.core.hooks.fp16_utils import force_fp32
from mmaction.models.utils.gather_loss import GatherLoss, VariedShapeGatherLoss
from einops import rearrange

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1)[:, None], b.norm(dim=-1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.matmul(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def cos_norm(a, eps=1e-8):
    if a is None:
        return a
    a_n = a.norm(dim=-1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm

@LOSSES.register_module()
class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.07, cos_sim=False):
        super().__init__()
        self.t = temperature
        self.use_cos_similarity = cos_sim
        self.allgather = GatherLoss.apply
        self.rank, self.world_size = get_dist_info()
        if self.use_cos_similarity:
            print("use cosine similarity")
        self.fp16_enabled = False

    @force_fp32()
    def forward(self, video_embd=None, text_embd=None, sim_mat=None):
        if sim_mat is None:           
            video_embd = self.allgather(video_embd, self.rank, self.world_size)
            text_embd = self.allgather(text_embd, self.rank, self.world_size)

            # video_embd shape: B x D
            # text_embd  shape: B x D
            if self.use_cos_similarity:
                x = sim_matrix(video_embd, text_embd) / self.t
            else:
                video_embd = F.normalize(video_embd, dim=-1)
                text_embd = F.normalize(text_embd, dim=-1)
                x = torch.matmul(video_embd, text_embd.t()) / self.t
            "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        else:
            x = sim_mat
        
        i_logsm = F.log_softmax(x, dim=1)
        j_logsm = F.log_softmax(x.t(), dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j

@LOSSES.register_module()
class DualSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.07, cos_sim=False, temp=1000):
        super().__init__()
        self.t = temperature
        self.use_cos_similarity = cos_sim
        self.allgather = GatherLoss.apply
        self.rank, self.world_size = get_dist_info()
        if self.use_cos_similarity:
            print("use cosine similarity")
        self.fp16_enabled = False
        self.temp = temp

    @force_fp32()
    def forward(self, video_embd=None, text_embd=None, sim_mat=None):
        if sim_mat is None:           
            video_embd = self.allgather(video_embd, self.rank, self.world_size)
            text_embd = self.allgather(text_embd, self.rank, self.world_size)

            # video_embd shape: B x D
            # text_embd  shape: B x D
            if self.use_cos_similarity:
                x = sim_matrix(video_embd, text_embd) / self.t
            else:
                video_embd = F.normalize(video_embd, dim=-1)
                text_embd = F.normalize(text_embd, dim=-1)
                x = torch.matmul(video_embd, text_embd.t()) / self.t
            "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        else:
            x = sim_mat
        
        i_sim_mat = x * F.softmax(x/self.temp, dim=0)*len(x) 
        j_sim_mat = x.t() * F.softmax(x.t()/self.temp, dim=0)*len(x.t()) 

        i_logsm = F.log_softmax(i_sim_mat, dim=1)
        j_logsm = F.log_softmax(j_sim_mat, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j
