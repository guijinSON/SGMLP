import torch 
import torch.nn as nn 
import torch.nn.functional as F


class gMLPBLOCK_CLS(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len):
        super(gMLPBLOCK_CLS,self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.channel_proj_i = nn.Sequential(nn.Linear(d_model,d_ffn),nn.GELU())
        self.channel_proj_ii = nn.Sequential(nn.Linear(d_ffn,d_model),nn.GELU())
        self.sgu = SpatialGatingUnit_CLS(d_ffn,seq_len)

    def forward(self,x):
        residual = x
        x = self.layer_norm(x)
        x = self.channel_proj_i(x)
        x = self.sgu(x)
        x = self.channel_proj_ii(x)
        return residual + x

class SpatialGatingUnit_CLS(nn.Module):
    def __init__(self,d_ffn,seq_len):
        super(SpatialGatingUnit_CLS,self).__init__()
        self.layer_norm = nn.LayerNorm(d_ffn)
        self.spatial_cls = nn.Linear(d_ffn,1)
        self.spatial_proj_i = nn.Conv1d(seq_len,seq_len,1)
        self.spatial_proj_ii = nn.Conv1d(seq_len,seq_len,1)
        nn.init.constant_(self.spatial_proj_i.bias, -1.0)
        nn.init.constant_(self.spatial_proj_ii.bias, 1.0)
        self.gelu = nn.GELU()

    def forward(self,x):
        residual = x #학습이 느려질 여지가 될 수도 residual / 파라미터 증가 
        x = self.layer_norm(x)
        cls  = x[:,0]
        cls  = torch.tanh(self.spatial_cls(cls))
        cls_idx = [0 if _<0 else 1 for _ in cls ] 
        output = [self.spatial_proj_i(torch.unsqueeze(x[n],0)) if idx==0 else self.spatial_proj_ii(torch.unsqueeze(x[n],0)) for n,idx in enumerate(cls_idx)]
        output = torch.squeeze(torch.stack(output)) + residual
        output = self.gelu(output)
        return output