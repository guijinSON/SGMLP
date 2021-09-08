from src.pretrain import PreTrainTrainer
from configs import parser
import os 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

trainer = PreTrainTrainer()
print('loaded')
trainer.pretrain()

#if __name__ ==  '__main__':
#    world_size = 4
#    mp.spawn(trainer.pretrain,args = (world_size, ), nprocs = world_size, join=True )
