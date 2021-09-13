from model.model import build_model
import torch 
import torch.nn as nn

class OneSentClassificationHead(nn.Module):
  def __init__(self,vocab_size,d_model,d_ffn, maxlen,layers,weight_path,device):
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
    x = self.pooler(x)
    x = self.projection(x)
    return x
    
  def load_model(self):
    model = build_model(self.vocab_size,self.d_model,self.d_ffn,self.maxlen,self.layers,output_loogits=False)
    weight = torch.load(self.weight_path,map_location=torch.device(self.device))
    model_weight = {}
    for key,val in weight.items():
      model_weight[key[7:]] = val
    model.load_state_dict(model_weight)
    return model 
    
