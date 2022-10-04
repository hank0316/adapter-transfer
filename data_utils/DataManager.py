import torch
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset, load_metric
from transformers import EvalPrediction

from data_utils.Dataset import GLUEDataset

class DatasetManager:
    def __init__(self, task_name, tokenizer='roberta-base', size=1000, data_seed=316):
        self.task_name = task_name
        self.num_labels = {'rte' : 2, 'stsb' : 1, 'mrpc' : 2, 'cola' : 2,
                           'mnli' : 3, 'sst2' : 2, 'qqp' : 2, 'qnli' : 2}
        self.data_seed = data_seed
        if self.task_name in ['rte', 'stsb', 'mrpc', 'cola', 'mnli', 'sst2', 'qqp', 'qnli']:
            self.raw_set = load_dataset('glue', self.task_name)
            self.data = GLUEDataset(self.raw_set, task_name=self.task_name, tokenizer=tokenizer, train_size=size,
                                      data_seed=self.data_seed)
            self.metric = load_metric('glue', self.task_name)
        else:
            raise NotImplementedError
    
    def getDataSplit(self, split='train'):
        if split == 'train':
            return self.data['train']
        elif split == 'eval':
            try:
                return self.data['validation']
            except:
                print('Using mnli matched!')
                return self.data['validation_matched']
        elif split == 'test':
            try:
                return self.data['test']
            except:
                return self.data['test_matched']

        else:
            return None
    
    def getMetric(self):
        def compute_with_argmax(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            return self.metric.compute(predictions=preds, references=p.label_ids)

        def compute_with_squeeze(p: EvalPrediction):
            preds = np.squeeze(p.predictions)
            labels = np.squeeze(p.label_ids)
            return self.metric.compute(predictions=preds, references=labels)
        
        if self.task_name in ['stsb']:
            return compute_with_squeeze
        elif self.task_name in ['rte', 'mrpc', 'cola', 'mnli', 'sst2', 'qqp', 'qnli']:
            return compute_with_argmax
        else:
            raise NotImplementedError

    def getCriteria(self):
        pass
    
    def getNumLabels(self):
        return self.num_labels[self.task_name]
