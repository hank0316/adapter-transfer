import os
import json
import torch
import numpy as np
import random
import torch.nn.functional as F

from typing import List, Union
from transformers import (
    RobertaConfig, 
    RobertaAdapterModel, 
    BertConfig, 
    BertAdapterModel, 
    AutoTokenizer, 
    DataCollatorForTokenClassification, 
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.optim import AdamW
from tqdm import tqdm

from data_utils.MTL_Manager import MTL_Manager


def trainMTL(tasks: List[str], load_if_exists, **kwargs):
    same_seeds(42)

    if 'config' in kwargs:
        config = kwargs["config"]
    else:
        config = default_config()
        print("[trainTransfer]: Using default config.")
        print(json.dumps(config, indent=4))

    # Configurations
    model_config = config['model']
    trainer_config = config['trainer']

    # DataManager
    manager = MTL_Manager(tasks, tokenizer=model_config['name'])
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
    # model
    model = init_model(model_config, tasks, manager)
	# Dataset, Dataloader
    train_sets, eval_sets = manager.getDataSplit('train'), manager.getDataSplit('eval')
    train_loaders = {
        task: DataLoader(
            train_sets[task], 
            batch_size=config['exp_setup']['mtl_batch_size'],
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True
        )
        for task in train_sets.keys()
    }
    eval_loaders = {
        task: DataLoader(
            eval_sets[task], 
            batch_size=config['exp_setup']['mtl_batch_size'],
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True
        )
        for task in eval_sets.keys()
    }
    train_iterators = {
        task: iter(train_loaders[task]) 
        for task in train_loaders.keys()
    }
    # Optimizer, Learning Rate Scheduler
    optimizer = AdamW(model.parameters(), lr=trainer_config['learning_rate'], weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=trainer_config['train_step'])
    model.train_adapter('adapter')
    model.to("cuda")
	
    records = []
    # Training Loop
    progress_bar = tqdm(range(trainer_config['train_step']), desc="Train", unit="step", initial=0)
    for step in progress_bar:
        model.train()
        loss, acc = 0.0, 0.0
        for task in tasks:
            # get data
            try:
                batch = next(train_iterators[task])
            except:
                train_iterators = {task: iter(train_loaders[task]) for task in train_loaders.keys()}
                batch = next(train_iterators[task])
            # To gpu
            for k, v in batch.items():
                batch[k] = v.to('cuda')
            
            labels, input_ids, attention_mask = batch['labels'], batch['input_ids'], batch['attention_mask']
            output = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                head=task,
                labels=labels
            )
            loss += output.loss
        
        progress_bar.set_postfix({'loss': loss.item()})
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if (step + 1) % trainer_config['eval_step'] == 0:
            # progress_bar.close()
            model.eval()
            eval_results = {}
            for eval_task in tasks:
                x, y = np.array([]), np.array([])
                eval_results[eval_task] = 0
                metric = manager.getMetric()[eval_task]
                eval_progress_bar = tqdm(eval_loaders[eval_task], desc=f'Eval {eval_task}', unit=' step')
                for batch in eval_progress_bar:
                    for k, v in batch.items():
                       batch[k] = v.to('cuda')

                    labels, input_ids, attention_mask = batch['labels'], batch['input_ids'], batch['attention_mask']
                    output = model(input_ids=input_ids, attention_mask=attention_mask, head=eval_task)
                    preds = output.logits

                    if eval_task != 'stsb':
                        # a classification task
                        preds = F.softmax(preds, dim=-1)
                        preds = preds.argmax(dim=-1)
                    else:
                        # regression task
                        preds, _ = preds.max(dim=-1)
                    
                    preds = preds.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                
                    if x.size != 0:
                        x = np.concatenate((x, preds))
                        y = np.concatenate((y, labels))
                    else:
                        x = preds
                        y = labels
                # Compute Metric
                eval_results[eval_task] = computeMetric(eval_task, np.stack(x), np.stack(y))
                eval_progress_bar.close()
            print('\n')
            print(json.dumps(eval_results, indent=4))
            records.append(eval_results)
    
    # Write to file
    progress_bar.close()
    with open(f'mtl_log/{"-".join(tasks)}.log', 'w') as log_file:
        step = 250
        for record in records:
            print(f'step: {step}', file=log_file)
            print(json.dumps(record, indent=4), file=log_file)
            print('-'*10, file=log_file)
            step += trainer_config['eval_step']

# Fix random seed for reproducibility
def same_seeds(seed):
	  torch.manual_seed(seed)
	  if torch.cuda.is_available():
		    torch.cuda.manual_seed(seed)
		    torch.cuda.manual_seed_all(seed)
	  np.random.seed(seed)
	  random.seed(seed)
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True



def init_model(model_config: dict, tasks: List[str], manager: MTL_Manager):
    # Load Model
    if 'roberta' in model_config['name']:
        pretrain_config = RobertaConfig.from_pretrained(model_config["name"])
        model = RobertaAdapterModel.from_pretrained(
            model_config["name"],
            config=pretrain_config
        )
    elif 'bert' in model_config['name']:
        pretrain_config = BertConfig.from_pretrained(model_config["name"])
        model = BertAdapterModel.from_pretrained(
            model_config["name"],
            config=pretrain_config
        )
    else:
        raise
    # Add adapter, heads
    model.add_adapter('adapter')
    num_labels = manager.getNumLabels()
    for task in tasks:
        model.add_classification_head(task, num_labels[task])
    return model


def collate_fn(samples: List[dict]):
    instance = defaultdict(list)
    for sample in samples:
        for k, v in sample.items():
            instance[k].append(v)
    for k, v in instance.items():
        instance[k] = torch.stack(v)
    
    return instance


def computeMetric(task: str, x, y):
    from scipy.stats import spearmanr
    from datasets import load_metric
    if task in ['rte', 'mrpc']:
        return {'acc': (x == y).mean()}
    elif task == 'stsb':
        return {'spearmanr': spearmanr(x, y).correlation}
    elif task == 'cola':
        metric = load_metric('matthews_correlation')
        return {"matthews": metric.compute(references=y, predictions=x)['matthews_correlation']}
    else:
        raise NotImplementedError