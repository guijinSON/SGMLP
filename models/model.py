import torch 
import torch.nn as nn 
import torch.nn.functional as F
import csv
from models.embedding import TransformerEmbedding
from models.layer import gMLPBLOCK



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

class MaskedLanguageModelingHead(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super(MaskedLanguageModelingHead,self).__init__()
        self.linear_layer = nn.Linear(model_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, encoder_output):
        # mask_position = [bs, tgt_size(15% of sent)]
        mlm_prediction = self.softmax(self.linear_layer(encoder_output)) # [bs,sl,vocab_size]
        
        return mlm_prediction

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
    

def build_model(num_tokens, d_model, d_ffn, seq_len, num_layers,output_logits=False,evaluate=False):
    
    model = gMLP_LanguageModel(num_tokens,d_model,d_ffn,seq_len,num_layers,output_logits,evaluate)
    #if torch.cuda.is_available():
    #    model = model.cuda()
    return model


class OneSentClassificationHead(nn.Module):
  def __init__(self,vocab_size,d_model,d_ffn, maxlen,layers,weight_path,device):
    super(OneSentClassificationHead,self).__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.d_ffn = d_ffn
    self.layers = layers
    self.maxlen = maxlen
    self.weight_path = weight_path
    self.device = device
    
    self.model = self.load_model()
    
    self.pooler = nn.Sequential(nn.Linear(self.d_model,self._model),
                                nn.Tanh())
    self.projection = nn.Sequential(nn.Dropout(),
                                    nn.Linear(self.d_model,1,bias=False))
    
  def forward(self,x):
    x = self.model(x)[:,0]
    x = self.pooler(x)
    x = self.projection(x)
    return x
    
  def load_model(self):
    model = build_model(self.vocab_size,self.d_model,self.d_ffn,self.maxlen,self.layers,output_logits=False)
    weight = torch.load(self.weight_path,map_location=torch.device(self.device))
    model_weight = {}
    for key,val in weight.items():
      model_weight[key[7:]] = val
    model.load_state_dict(model_weight)
    return model 
