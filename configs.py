import os
import sys
import argparse
import random
import torch
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--training_steps', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--evaluate',type=int, default=1000)
parser.add_argument('--checkpoint_model',type=int, default=100000)


parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_ffn', type=int, default=2048)
parser.add_argument('--num_layers', type=int, default=12)
parser.add_argument('--drop_prob',type=float, default=0.1)

parser.add_argument('--init_lr',type=float, default=2e-5)
parser.add_argument('--warm_up',type=int, default=10000)

parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--vocab_size',type=int, default=50265)
# trainer
parser.add_argument('--pad_token',type=int, default=1)
parser.add_argument('--mask_token',type=int, default=50264)
parser.add_argument('--mask_prob',type=int, default=0.3)

parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--weight_path', type=str, default=None)
parser.add_argument('--best_pretrain_epoch',type=int, default=1)

def get_config():
    return parser

def set_random_fixed(seed_num):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    np.random.seed(seed_num)

def get_path_info():
    cur_path = os.getcwd()
    weight_path = os.path.join(cur_path,'weights')
    final_model_path = os.path.join(cur_path,'final_results')
    
    return cur_path, weight_path, final_model_path