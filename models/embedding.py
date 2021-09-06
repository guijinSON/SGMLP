import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, model_dim):
        super(TokenEmbedding, self).__init__(vocab_size, model_dim, padding_idx=0)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, model_dim).float()
        pe.requires_grad = False
        
        pos = torch.arange(0,max_len).float().unsqueeze(dim=1)
        divterm = (torch.arange(0,model_dim,step=2).float() * -(math.log(10000.0) / model_dim)).exp()
        
        # pe = (1, sequence_length, hidden_size)
        pe[:, 0::2] = torch.sin(pos * divterm)
        pe[:, 1::2] = torch.cos(pos * divterm)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self, tensor):
        
        # (1, sequence_length, hidden_size)
        return self.pe[:, :tensor.size(1)]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, max_len, drop_prob):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, model_dim)
        self.pos_emb = PositionalEncoding(model_dim, max_len)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, tensor):
        tok_emb = self.tok_emb(tensor)
        pos_emb = self.pos_emb(tensor)
        
        return self.dropout(tok_emb + pos_emb)