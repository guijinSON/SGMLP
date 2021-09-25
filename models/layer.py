import torch 
import torch.nn as nn 
import torch.nn.functional as F

class SpatialGatingUnit_Sigmoid(nn.Module):
    def __init__(self,d_ffn,seq_len,evaluate,wr):
        super(SpatialGatingUnit_Sigmoid,self).__init__()
        self.layer_norm = nn.LayerNorm(d_ffn)
        self.spatial_cls = nn.Sequential(nn.Linear(d_ffn,10),nn.GELU(),
                                        nn.Linear(10,1),nn.Sigmoid())
        self.spatial_proj_i = nn.Conv1d(seq_len,seq_len,1)
        self.spatial_proj_ii = nn.Conv1d(seq_len,seq_len,1)

        nn.init.constant_(self.spatial_proj_i.bias, -1.0)
        nn.init.constant_(self.spatial_proj_ii.bias, 1.0)
        self.gelu = nn.GELU()
        self.wr = wr
        self.evaluate = evaluate
        
    def forward(self,x):
        residual = x.clone() #학습이 느려질 여지가 될 수도 residual / 파라미터 증가 
        x = self.layer_norm(x)
        cls  = x[:,0]
        cls  = self.spatial_cls(cls)
        cls_idx = torch.round(cls).type(torch.LongTensor)
        if self.evaluate:
            self.wr.writerow(cls_idx)

        output = [self.spatial_proj_i(torch.unsqueeze(x[n],0)) if idx==0 else self.spatial_proj_ii(torch.unsqueeze(x[n],0)) for n,idx in enumerate(cls_idx)]
        output = torch.squeeze(torch.stack(output)) 
        output = self.gelu(output+residual) 
        return output

class SpatialGatingUnit_Sigmoid_Extended(nn.Module):
    def __init__(self,d_ffn,seq_len,evaluate,wr):
        super(SpatialGatingUnit_Sigmoid_Extended,self).__init__()
        self.layer_norm = nn.LayerNorm(d_ffn)
        self.spatial_cls = nn.Sequential(nn.Linear(d_ffn,10),nn.GELU(),
                                        nn.Linear(10,1),nn.Sigmoid())
        self.spatial_proj_i = nn.Conv1d(seq_len,seq_len,1)
        self.spatial_proj_ii = nn.Conv1d(seq_len,seq_len,1)

        self.attn_extention = nn.MultiheadAttention(d_ffn, 2,batch_first=True)

        nn.init.constant_(self.spatial_proj_i.bias, -1.0)
        nn.init.constant_(self.spatial_proj_ii.bias, 1.0)
        self.gelu = nn.GELU()
        self.wr = wr
        self.evaluate = evaluate
        
    def forward(self,x):
        residual = x.clone() #학습이 느려질 여지가 될 수도 residual / 파라미터 증가 
        x = self.layer_norm(x)
        cls  = x[:,0]
        cls  = self.spatial_cls(cls)
        cls_idx = torch.round(cls).type(torch.LongTensor)
        if self.evaluate:
            self.wr.writerow(cls_idx)

        output = [self.spatial_proj_i(torch.unsqueeze(x[n],0)) if idx==0 else self.spatial_proj_ii(torch.unsqueeze(x[n],0)) for n,idx in enumerate(cls_idx)]
        output = torch.squeeze(torch.stack(output)) 
        output = self.gelu(output+residual) + self.attn_extention(residual,residual,residual)
        return output

class SpatialGatingUnit_multi(nn.Module):
    def __init__(self,d_ffn,seq_len,evaluate,n):
        super(SpatialGatingUnit_multi,self).__init__()
        self.layer_norm = nn.LayerNorm(d_ffn)
        self.spatial_cls = nn.Sequential(nn.Linear(int(d_ffn/24),10),nn.GELU(),
                                        nn.Conv1d(seq_len,1,1),nn.GELU(),
                                        nn.Linear(10,1),nn.Sigmoid())
        self.spatial_proj_i = nn.Conv1d(seq_len,seq_len,1)
        self.spatial_proj_ii = nn.Conv1d(seq_len,seq_len,1)

        nn.init.constant_(self.spatial_proj_i.bias, -1.0)
        nn.init.constant_(self.spatial_proj_ii.bias, 1.0)
        self.gelu = nn.GELU()
        self.n = n
        self.evaluate = evaluate
        self.d_ffn = d_ffn 
        
    def forward(self,x):
        residual = x.clone() #학습이 느려질 여지가 될 수도 residual / 파라미터 증가 
        x = self.layer_norm(x)
        cls  = x[:,:,int(self.d_ffn/24)*self.n:(self.n+1)*int(self.d_ffn/24)]

        cls  = self.spatial_cls(cls)

        cls_idx = torch.round(torch.squeeze(cls)).type(torch.LongTensor)
        if self.evaluate:
            self.wr.writerow(cls_idx)

        output = [self.spatial_proj_i(torch.unsqueeze(x[n],0)) if idx==0 else self.spatial_proj_ii(torch.unsqueeze(x[n],0)) for n,idx in enumerate(cls_idx)]
        output = torch.squeeze(torch.stack(output)) 
        output = self.gelu(output+residual)

class gMLPBLOCK(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len,evaluate,wr=None):
        super(gMLPBLOCK,self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.channel_proj_i = nn.Sequential(nn.Linear(d_model,d_ffn),nn.GELU())
        self.channel_proj_ii = nn.Sequential(nn.Linear(d_ffn,d_model),nn.GELU())
        self.sgu = SpatialGatingUnit_Sigmoid(d_ffn,seq_len,evaluate,wr)

    def forward(self,x):
        residual = x.clone()
        x = self.layer_norm(x)
        x = self.channel_proj_i(x)
        x = self.sgu(x)
        x = self.channel_proj_ii(x)
        return x + residual

class gMLPBLOCK_Extended(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len,evaluate,wr=None):
        super(gMLPBLOCK_Extended,self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.channel_proj_i = nn.Sequential(nn.Linear(d_model,d_ffn),nn.GELU())
        self.channel_proj_ii = nn.Sequential(nn.Linear(d_ffn,d_model),nn.GELU())
        self.sgu = SpatialGatingUnit_Sigmoid_Extended(d_ffn,seq_len,evaluate,wr)

    def forward(self,x):
        residual = x.clone()
        x = self.layer_norm(x)
        x = self.channel_proj_i(x)
        x = self.sgu(x)
        x = self.channel_proj_ii(x)
        return x + residual

class multi_gMLPBLOCK(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len,evaluate,n):
        super(multi_gMLPBLOCK,self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.channel_proj_i = nn.Sequential(nn.Linear(d_model,d_ffn),nn.GELU())
        self.channel_proj_ii = nn.Sequential(nn.Linear(d_ffn,d_model),nn.GELU())
        self.sgu = SpatialGatingUnit_multi(d_ffn,seq_len,evaluate,n)

    def forward(self,x):
        residual = x.clone()
        x = self.layer_norm(x)
        x = self.channel_proj_i(x)
        x = self.sgu(x)
        x = self.channel_proj_ii(x)
        return x + residual
