from datasets import load_dataset
from data.tokenizer import load_tokenizer
from torch.utils.data import DataLoader,DistributedSampler
from datasets import interleave_datasets
import numpy as np
import random
from configs import parser
from datasets import concatenate_datasets, load_dataset


#args = parser.parse_args()

def load_dataloader(max_length=128,batch_size=64,rank=None,world_size=None):
    train_allenc4_dataset = load_dataset("allenai/c4",'en', split='train', streaming=True).take(256*20000)
    train_c4_dataset = load_dataset('c4', "en", split='train', streaming=True).take(256*20000)
   # train_wiki_dataset = load_dataset("wikicorpus", "raw_en",split='train',streaming=True)

    #bookcorpus = load_dataset("bookcorpus", split="train")
    #wiki = load_dataset("wikipedia", "20200501.en", split="train")

    val_dataset = load_dataset("c4",'en', split='validation',streaming=True)#,cache_dir='/raid/nlp/data')#, streaming=True)

    train_dataset = interleave_datasets([train_allenc4_dataset,train_c4_dataset])
    val_dataset = val_dataset.take(1000)
    
    tokenizer = load_tokenizer()
    train_dataset = tokenize_dataset(train_dataset,tokenizer,max_length)
    val_dataset   = tokenize_dataset(val_dataset,tokenizer,max_length)
    
    #train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True)
    #val_sampler = DistributedSampler(val_dataset,num_replicas=world_size,rank=rank,shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)#, sampler = train_sampler, shuffle=False, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)#, sampler = val_sampler, shuffle=False, pin_memory=True)
    return train_loader,val_loader

def tokenize_dataset(dataset,tokenizer,max_length):
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"],max_length=max_length, padding='max_length',truncation=True,return_tensors='np')['input_ids'])
    masked_dataset = tokenized_dataset.map(mask_batch)
    torch_tokenized_dataset = masked_dataset.with_format("torch")
    return torch_tokenized_dataset

def mask_sent(sent,mask_token,mask_prob):
    sep_idx = np.where(sent==2)
    sent_len = int(sep_idx[0] - 1)
    sent[1:sent_len] = [mask_token if random.random() < mask_prob else tok for tok in sent[1:sent_len]]
    return sent

def mask_batch(batch):
    mask_token = args.mask_token
    mask_prob  = args.mask_prob
    label = batch.copy()
    return np.stack([mask_sent(sent,mask_token,mask_prob) for sent in batch]),label

def load_glue_dataset(task,split,max_length=128):
    dataset = load_dataset('glue', task, split=split)
    tokenizer = load_tokenizer()
    keys = [key for key in dataset.column_names if key not in ['label','idx']]
    for key in keys:
        tokenized = tokenizer(dataset[key],max_length=max_length,truncation=True, padding='max_length')
        input_ids = f'{key}_input_ids'
        dataset = dataset.add_column(input_ids, tokenized['input_ids'])
    column = [_ for _ in dataset.column_names if _ not in keys and _ != 'idx']
    dataset.set_format(type='torch',columns=column)
    return dataset
