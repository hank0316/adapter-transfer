import torch
import datasets
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, data_split, task_name=None, tokenizer='roberta-base', train_size=1000, data_seed=316):
        self.data = data_split
        self.task_name = task_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.data_seed = data_seed
        self.data['train'] = self.subset(self.data['train'], train_size)
        self.data = self.data.map(self.encode_batch, batched=True)
        self.data = self.data.rename_column('label', 'labels')
        self.data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
        
    def encode_batch(self, batch):
        if self.task_name in ['rte', 'stsb', 'mrpc']:
            return self.tokenizer(batch['sentence1'], batch['sentence2'], padding='max_length')
        elif self.task_name in ['cola']:
            return self.tokenizer(batch['sentence'], padding='max_length')
        else:
            raise NotImplementedError
    
    def subset(self, data_to_subset, size):
        torch.manual_seed(self.data_seed)
        indices = torch.randperm(len(data_to_subset))[: min(len(data_to_subset), size)]
        return datasets.Dataset.from_dict(data_to_subset[indices])
