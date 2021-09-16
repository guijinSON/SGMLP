from models.model import OneSentClassificationHead
import torch 
import torch.nn as nn
from utils import load_optimizer,load_lossfn
from datasets import load_metric
import wandb
from data.dataloader import load_glue_dataset
from tqdm import tqdm

class FinetuneTrainer():
  def __init__(self,task,vocab_size,d_model,d_ffn, maxlen,layers,weight_path,device,lr=5e-5,batch_size=64,evaluate=50):
    self.task = task.lower()
    self.device = device
    if self.task in ['cola','sst2','stsb']:
      self.model = OneSentClassificationHead(vocab_size,d_model,d_ffn, maxlen,layers,weight_path,device).to(self.device)
      
      
    self.loss_fn = load_lossfn(self.task)
    self.metric = load_metric('glue',self.task)
    self.optimizer = load_optimizer(self.model.parameters(),learning_rate=lr)
    
    train_dataset = load_glue_dataset(self.task,'train',max_length=maxlen)
    val_dataset = load_glue_dataset(self.task,'validation',max_length=maxlen)
    
    self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle =True)
    
      
  def finetune(self):
    wandb.init()
    self.model.train()
    step =0
    for epoch in range(10):
      running_loss = 0.0
      for batch in tqdm(self.train_dataloader):
        x = batch['sentence_input_ids'].to(self.device)
        label = batch['label'].to(torch.float).to(self.device)

        self.optimizer.zero_grad()

        pred = self.model(x).squeeze()

        loss = self.loss_fn(pred,label)
        loss.backward()

        running_loss += loss.item()

        #self.model.eval()
        validation_loss = 0.0
        step+=1
        if step % evaluate == 0:
          with torch.no_grad():
              for batch in self.val_dataloader:
                  x = batch['sentence_input_ids'].to(self.device)
                  label = batch['label'].to(self.device)

                  pred = self.model(x).squeeze()
                  loss = self.loss_fn(pred,label.to(torch.float))

                  pred = torch.round(pred).to(torch.int)
                  validation_loss += loss.item()
                  self.metric.add_batch(predictions=pred,references=label)

              wandb.log({"Train Loss": (running_loss/len(self.train_dataloader)),'metric':self.metric.compute(), "Validation Loss":(validation_loss/len(self.val_dataloader))})
         # self.model.train()
        running_loss = 0.0
      torch.save(self.model.state_dict(),f'{self.task}_{epoch}.pth')
    
