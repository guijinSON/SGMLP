from configs import get_config
from data.dataloader import load_dataloader
from models.model import build_model
import torch
import os
import torch.distributed as dist
from utils import load_optimizer , load_lossfn , ScheduledOptim
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

class PreTrainTrainer():
    def __init__(self):
        parser = get_config()
        self.args = parser.parse_args()
        self.train_loader, self.val_loader = load_dataloader(self.args.seq_len,self.args.batch_size)
        self.model = build_model(self.args.vocab_size, self.args.d_model, self.args.d_ffn, self.args.seq_len, self.args.num_layers,output_logits=True)
        self.optimizer = load_optimizer(self.model.parameters(),self.args.init_lr)
        self.scheduler = ScheduledOptim(self.optimizer,self.args.d_model,self.args.warm_up)
        self.loss_fn = load_lossfn(self.args.pad_token)

        
    def pretrain(self):

        #self.setup(rank,world_size)
        #self.model = self.model.to(rank)
        #self.model = DDP(self.model, device_ids=[rank],output_device=rank)
        self.model = self.model.to('cuda:0')
        self.model = torch.nn.DataParallel(self.model,device_ids=[0,1,2])

        #self.train_loader, self.val_loader = load_dataloader(self.args.seq_len,self.args.batch_size,rank,world_size)
        
        wandb.init(group='DDP')
        
        step = 0
        running_loss = 0.0
        
        #self.train_loader.sampler.set_epoch(step)
        for epoch in range(self.args.epochs):
            for batch,label in self.train_loader:

                batch = torch.squeeze(batch).to('cuda:0')#.to(rank)
                label = label.flatten().to('cuda:0')#.to(rank)

                self.optimizer.zero_grad()

                pred = self.model(batch)
                pred = pred.contiguous().view(-1,pred.shape[-1])

                loss = self.loss_fn(pred,label)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step_and_update_lr()

                step += 1
                if step % self.args.evaluate == 0:
                    self.model.eval()
                    with torch.no_grad():
                        validation_loss = 0.0
                        for batch,label in self.val_loader:
                            batch = torch.squeeze(batch).to('cuda:0')#.to(rank)
                            label = label.flatten().to('cuda:0')#.to(rank)

                            pred = self.model(batch)
                            pred = pred.contiguous().view(-1,pred.shape[-1])

                            loss = self.loss_fn(pred,label)
                            validation_loss += loss.item()

                    wandb.log({"Training Loss": (running_loss/self.args.evaluate),
                                'Valiation Losss':(validation_loss/1000*100)})
                    running_loss = 0.0
                    self.model.train()

                if step % self.args.checkpoint_model == 0:
                    torch.save(self.model.state_dict(),f'weights/model_{step}.pth')

                if step == self.args.training_steps:
                    torch.save(self.model.state_dict(),'weights/model_fin.pth')
                    #self.cleanup()
                    break 

    def setup(self,rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()

