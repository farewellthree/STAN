import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmaction.registry import MODELS
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

@MODELS.register_module()
class NormSoftmaxLoss(nn.Module):
    def forward(self, video_embd=None, text_embd=None, sim_mat=None, scale=None):
        if sim_mat is None:           
            x = sim_matrix(video_embd, text_embd) 
            x = x if scale is None else x * scale
        else:
            x = sim_mat if scale is None else sim_mat * scale
        
        i_logsm = F.log_softmax(x, dim=1)
        j_logsm = F.log_softmax(x.t(), dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j