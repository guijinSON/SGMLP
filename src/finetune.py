from models.model import OneSentClassificationHead
import torch 
import torch.nn as nn
from utils import load_optimizer,load_lossfn
from datasets import load_metric
import wandb
from data.dataloader import load_glue_dataset
from tqdm import tqdm

class FintuneTrainer():
  def __init__(self,task,vocab_size,d_model,d_ffn, maxlen,layers,weight_path,device,lr=5e-5,batch_size=64):
    self.task = task.lower()
    self.device = device
    if self.task in ['cola','sst2','sstb']:
      self.model = OneSentClassificationHead(vocab_size,d_model,d_ffn, maxlen,layers,weight_path,device)
      
      
    self.loss_fn = load_lossfn(self.task)
    self.metric = load_metric('glue',self.task)
    self.optimizer = load_optimizer(self.model.parameters(),lr=lr)
    
    train_data = load_glue_dataset(self.task,'train',max_length=maxlen)
    val_data = load_glue_dataset(self.task,'validation',max_length=maxlen)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle =True)
    
      
  def finetune(self):
    wandb.init()
    self.model.train()
    for epoch in range(10):
      running_loss = 0.0
      for batch in tqdm(self.train_dataloader):
         x = batch['input_ids'].to(self.device)
            label = batch['label'].to(torch.float).to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(x).squeeze()

            loss = self.loss_fn(pred,label)
            loss.backward()

            running_loss += loss.item()

      self.model.eval()
      validation_loss = 0.0
      with torch.no_grad():
          for batch in self.val_loader:
              x = batch['input_ids'].to(self.device)
              label = batch['label'].to(self.device)

              pred = self.model(x).squeeze()
              loss = self.loss_fn(pred,label.to(torch.float))

              pred = torch.round(pred).to(torch.int)
              self.metric.add_batch(predictions=pred,references=label)

          wandb.log({"Train Loss": (running_loss/evaluate),'metric':self.metric.compute(), "Validation Loss":(validation_loss/len(self.val_loader))})
      self.model.train()
      running_loss = 0.0
      torch.save(self.model.state_dict(),f'{self.task}_{epoch}.pth')
    
