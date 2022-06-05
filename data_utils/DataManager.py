import torch
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset, load_metric
from transformers import EvalPrediction

from data_utils.Dataset import CustomDataset

class DatasetManager:
    def __init__(self, task_name, tokenizer='roberta-base', size=1000, data_seed=316):
        self.task_name = task_name
        self.num_labels = {'rte' : 2, 'stsb' : 1, 'mrpc' : 2, 'cola' : 2}
        self.data_seed = data_seed
        if self.task_name in ['rte', 'stsb', 'mrpc', 'cola']:
            self.raw_set = load_dataset('glue', self.task_name)
            self.data = CustomDataset(self.raw_set, task_name=self.task_name, tokenizer=tokenizer, train_size=size,
                                      data_seed=self.data_seed)
        else:
            raise NotImplementedError
    
    def getDataSplit(self, split='train'):
        if split == 'train':
            return self.data['train']
        elif split == 'eval':
            return self.data['validation']
        elif split == 'test':
            return self.data['test']
        else:
            return None
    
    def getMetric(self):
        def compute_accuracy(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            return {"acc": (preds == p.label_ids).mean()}

        def compute_spearmanr(p: EvalPrediction):
            preds = np.squeeze(p.predictions)
            labels = np.squeeze(p.label_ids)
            return {"spearmanr": spearmanr(preds, labels).correlation}
        
        def compute_matthews(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            metric = load_metric('matthews_correlation')
            return {"matthews": metric.compute(references=p.label_ids, predictions=preds)['matthews_correlation']}
        
        if self.task_name in ['rte', 'mrpc']:
            return compute_accuracy
        elif self.task_name in ['stsb']:
            return compute_spearmanr
        elif self.task_name in ['cola']:
            return compute_matthews
        else:
            raise NotImplementedError

    def getCriteria(self):
        pass
    
    def getNumLabels(self):
        return self.num_labels[self.task_name]
