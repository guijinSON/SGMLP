from configs import parser
from data.dataloader import load_dataloader
from models.model import build_model
import torch
from utils import load_optimizer , load_lossfn , ScheduledOptim
import wandb

class PreTrainTrainer():
    def __init__(self,parser):
        self.args = parser.parse_args()

        self.train_loader, self.val_loader = load_dataloader(self.args.seq_len,self.args.batch_size)
        self.model = build_model(self.args.vocab_size, self.args.d_model, self.args.d_ffn, self.args.seq_len, self.args.num_layers,output_logits=True)

        self.optimizer = load_optimizer(self.model.parameters(),self.args.init_lr)
        self.scheduler = ScheduledOptim(self.optimizer,self.args.d_model,self.args.warm_up)
        self.loss_fn = load_lossfn(self.args.pad_idx)

    
    def pretrain(self):
        print('initalize wandb')
        wandb.init()
        step = 0
        running_loss = 0.0
        for batch,label in self.train_loader:
            batch = torch.squeeze(batch)
            label = label.flatten()

            self.optimizer.zero_grad()

            pred = self.model(batch)
            pred = pred.contiguous().view(-1,pred.shape[-1])

            loss = self.loss_fn(pred,label)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step_and_update_lr()

            step +=1
            if step % self.args.evaluate == 0:
                self.model.eval()
                with torch.no_grad():
                    validation_loss = 0.0
                    for batch,label in self.val_loader:
                        batch = torch.squeeze(batch)
                        label = label.flatten()

                        pred = self.model(batch)
                        pred = pred.contiguous().view(-1,pred.shape[-1])

                        loss = self.loss_fn(pred,label)
                        validation_loss += loss.item()

                wandb.log({"Training Loss": (running_loss/self.args.evaluate),
                            'Valiation Losss':(validation_loss/1000*100)})
                self.model.train()

            if step % self.args.checkpoint_model == 0:
                torch.save(self.model.state_dict(),f'/weights/model_{step}.pth')

            if step == self.args.training_steps:
                torch.save(self.model.state_dict(),'/weights/model_fin.pth')
                break 



