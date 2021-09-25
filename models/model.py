import torch 
import torch.nn as nn 
import torch.nn.functional as F
import csv
from models.embedding import TransformerEmbedding
from models.layer import gMLPBLOCK,multi_gMLPBLOCK,gMLPBLOCK_Extended,multi_gMLPBLOCK_Extended


class MaskedLanguageModelingHead(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super(MaskedLanguageModelingHead,self).__init__()
        self.linear_layer = nn.Linear(model_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, encoder_output):
        # mask_position = [bs, tgt_size(15% of sent)]
        mlm_prediction = self.softmax(self.linear_layer(encoder_output)) # [bs,sl,vocab_size]
        
        return mlm_prediction

class gMLP(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len,num_layers,evaluate):
        super(gMLP,self).__init__()
        self.evaluate = evaluate
        if self.evaluate:
            route = open('weights/routing.csv','w', newline='')
            wr = csv.writer(route)
            self.model = nn.Sequential(*[gMLPBLOCK(d_model,d_ffn,seq_len,evaluate,wr) for _ in range(num_layers)])
        else:
            self.model = nn.Sequential(*[gMLPBLOCK(d_model,d_ffn,seq_len,evaluate) for _ in range(num_layers)])
    def forward(self,x):
        if self.evaluate:
            x,csv = model(x)
            csv.close()
            return x,csv
        x = self.model(x)
        return x
    
class gMLP_Extended(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len,num_layers,evaluate):
        super(gMLP_Extended,self).__init__()
        self.evaluate = evaluate
        if self.evaluate:
            route = open('weights/routing.csv','w', newline='')
            wr = csv.writer(route)
            self.model = nn.Sequential(*[gMLPBLOCK_Extended(d_model,d_ffn,seq_len,evaluate,wr) for _ in range(num_layers)])
        else:
            self.model = nn.Sequential(*[gMLPBLOCK_Extended(d_model,d_ffn,seq_len,evaluate,_) for _ in range(num_layers)])
    def forward(self,x):
        if self.evaluate:
            x,csv = model(x)
            csv.close()
            return x,csv
        x = self.model(x)
        return x

class multi_gMLP(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len,num_layers,evaluate):
        super(multi_gMLP,self).__init__()
        self.evaluate = evaluate
        if self.evaluate:
            route = open('weights/routing.csv','w', newline='')
            wr = csv.writer(route)
            self.model = nn.Sequential(*[multi_gMLPBLOCK(d_model,d_ffn,seq_len,evaluate,wr) for _ in range(num_layers)])
        else:
            self.model = nn.Sequential(*[multi_gMLPBLOCK(d_model,d_ffn,seq_len,evaluate,_) for _ in range(num_layers)])
    def forward(self,x):
        if self.evaluate:
            x,csv = model(x)
            csv.close()
            return x,csv
        x = self.model(x)
        return x
    
class multi_gMLP_Extended(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len,num_layers,evaluate):
        super(multi_gMLP_Extended,self).__init__()
        self.evaluate = evaluate
        if self.evaluate:
            route = open('weights/routing.csv','w', newline='')
            wr = csv.writer(route)
            self.model = nn.Sequential(*[multi_gMLPBLOCK_Extended(d_model,d_ffn,seq_len,evaluate,wr) for _ in range(num_layers)])
        else:
            self.model = nn.Sequential(*[multi_gMLPBLOCK_Extended(d_model,d_ffn,seq_len,evaluate,_) for _ in range(num_layers)])
    def forward(self,x):
        if self.evaluate:
            x,csv = model(x)
            csv.close()
            return x,csv
        x = self.model(x)
        return x
        
class gMLP_LanguageModel(gMLP):
    def __init__(self,vocab_size, d_model, d_ffn, seq_len, num_layers,output_logits,evaluate):
        super().__init__(d_model,d_ffn,seq_len,num_layers,evaluate)
        self.embed = TransformerEmbedding(vocab_size,d_model,seq_len,0.1)
        self.output_logits = output_logits
        self.to_logits = MaskedLanguageModelingHead(vocab_size,d_model)

    def forward(self,x):
        embedding = self.embed(x)
        embedding = embedding
        output = self.model(embedding)
        if self.output_logits:
            output = self.to_logits(output)

        return output

class gMLP_LanguageModel_Extended(gMLP_Extended):
    def __init__(self,vocab_size, d_model, d_ffn, seq_len, num_layers,output_logits,evaluate):
        super().__init__(d_model,d_ffn,seq_len,num_layers,evaluate)
        self.embed = TransformerEmbedding(vocab_size,d_model,seq_len,0.1)
        self.output_logits = output_logits
        self.to_logits = MaskedLanguageModelingHead(vocab_size,d_model)

    def forward(self,x):
        embedding = self.embed(x)
        embedding = embedding
        output = self.model(embedding)
        if self.output_logits:
            output = self.to_logits(output)

        return output
    
class gMLP_multi_LanguageModel(multi_gMLP):
    def __init__(self,vocab_size, d_model, d_ffn, seq_len, num_layers,output_logits,evaluate):
        super().__init__(d_model,d_ffn,seq_len,num_layers,evaluate)
        self.embed = TransformerEmbedding(vocab_size,d_model,seq_len,0.1)
        self.output_logits = output_logits
        self.to_logits = MaskedLanguageModelingHead(vocab_size,d_model)

    def forward(self,x):
        embedding = self.embed(x)
        embedding = embedding
        output = self.model(embedding)
        if self.output_logits:
            output = self.to_logits(output)

        return output
    
class gMLP_multi_LanguageModel_Extended(multi_gMLP_Extended):
    def __init__(self,vocab_size, d_model, d_ffn, seq_len, num_layers,output_logits,evaluate):
        super().__init__(d_model,d_ffn,seq_len,num_layers,evaluate)
        self.embed = TransformerEmbedding(vocab_size,d_model,seq_len,0.1)
        self.output_logits = output_logits
        self.to_logits = MaskedLanguageModelingHead(vocab_size,d_model)

    def forward(self,x):
        embedding = self.embed(x)
        embedding = embedding
        output = self.model(embedding)
        if self.output_logits:
            output = self.to_logits(output)

        return output
    

    

def build_base_model(num_tokens=50265, d_model=768, d_ffn=2048, seq_len=128, num_layers=12,output_logits=False,evaluate=False,extended_version=False):
    if extended_version:
        model = gMLP_LanguageModel_Extended(num_tokens,d_model,d_ffn,seq_len,num_layers,output_logits,evaluate)
    else:   
        model = gMLP_LanguageModel(num_tokens,d_model,d_ffn,seq_len,num_layers,output_logits,evaluate)
    #if torch.cuda.is_available():
    #    model = model.cuda()
    return model


def build_large_model(num_tokens=50265, d_model=768, d_ffn=2400, seq_len=128, num_layers=24,output_logits=False,evaluate=False,extended_version=False):
    if extended_version:
        model = gMLP_multi_LanguageModel_Extended(num_tokens,d_model,d_ffn,seq_len,num_layers,output_logits,evaluate)
    else:   
        model = gMLP_multi_LanguageModel(num_tokens,d_model,d_ffn,seq_len,num_layers,output_logits,evaluate)
    #if torch.cuda.is_available():
    #    model = model.cuda()
    return model
    
