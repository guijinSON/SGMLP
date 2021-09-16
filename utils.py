import numpy as np
import time
import torch
import torch.optim as  optim 
import torch.nn as nn

def load_optimizer(param, learning_rate):
    return optim.AdamW(params=param, lr=learning_rate)

def load_lossfn(task='Pretrain',ignore_idx=None):
    if task == 'Pretrain':
        return nn.NLLLoss(ignore_index=ignore_idx)
    if task in ['cola','sst2','sstb']:
        return nn.BCEWithLogitsLoss()

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            
def compute_time(model,vocab_size = 50265, batch_size = 64, max_length = 128, device='cuda'):
    inputs = torch.randint(0,vocab_size,(batch_size, max_length))
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    i = 0
    time_spent = []
    while i < 100:
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    print('Avg execution time (ms): {:.3f}'.format(np.mean(time_spent)))

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
