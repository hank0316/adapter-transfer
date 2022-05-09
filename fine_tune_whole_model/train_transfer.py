import os
import json

from typing import List, Union
from transformers import (
    AutoConfig, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

from data_utils.DataManager import DatasetManager

def default_config():
    return {
        "trainer": {
            "train_step": 2000,
            "logging_step": 250,
            "eval_step": 250,
            "save_step": 250,
            "load_best_model_at_end": True,
            "overwrite_output_dir": True,
            "remove_unused_columns": False,
            "ckpt_path": "checkpoint-best",
            "logging_dir": 'log',
            "metric_for_best_model" : 'eval_'
        },
        "model": {
            "name": "roberta-base"
        }
    }


def trainSingleTransfer(
    task: str, 
    config: dict,
    prev_base_model: AutoModelForSequenceClassification,
    cur_dir: list,
    load_if_exists=True
):
    trainer_config = config['trainer']
    model_config = config['model']
    task_config = config['exp_setup']['tasks']

    save_path = f'{os.path.join(*cur_dir)}/{trainer_config["ckpt_path"]}'

    if load_if_exists == True and os.path.exists(save_path):
        model = AutoModelForSequenceClassification.from_pretrained(save_path)
        print(f'{save_path} found. Weights are loaded and skip training.')
        return

    print(f'{save_path} not found. Start training...')

    # Start Training
    data_manager = DatasetManager(task, tokenizer=model_config['name'])

    # Load Model
    if 'roberta' in model_config['name'] or 'bert' in model_config['name']:
        pretrain_config = AutoConfig.from_pretrained(
            model_config["name"], 
            num_labels=data_manager.getNumLabels()
        )
        print(f'[Info]: pretrain_config = {pretrain_config}')
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config["name"], 
            config=pretrain_config
        )

        if 'roberta' in model_config['name']:
            model.roberta = prev_base_model if prev_base_model is not None else model.roberta
        elif 'bert' in model_config['name']:
            model.bert = prev_base_model if prev_base_model is not None else model.bert
        else:
            raise NotImplementedError
        
        print(f'[Info]: model: {model}')
    else:
        raise NotImplementedError

    # Arguments for Trainer
    training_args = TrainingArguments(
        learning_rate=trainer_config['learning_rate'],
        max_steps=trainer_config['train_step'],
        save_steps=trainer_config['save_step'],

        eval_steps=trainer_config['eval_step'],
        evaluation_strategy='steps',

        logging_steps=trainer_config['logging_step'],
        logging_dir=trainer_config['logging_dir'],

        load_best_model_at_end=trainer_config['load_best_model_at_end'],
        metric_for_best_model=trainer_config['metric_for_best_model'] + task_config['metrics'][task],

        output_dir=f"{os.path.join(*cur_dir)}",
        overwrite_output_dir=trainer_config['overwrite_output_dir'],
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=trainer_config['remove_unused_columns'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_manager.getDataSplit('train'),
        eval_dataset=data_manager.getDataSplit('eval'),
        compute_metrics=data_manager.getMetric()
    )
    
    trainer.train()
    res = trainer.evaluate()
    with open(f'{os.path.join(*cur_dir)}/result.json', 'w') as f:
        json.dump(res, f, indent=4)

    # Save result
    os.mkdir(save_path)
    model.save_pretrained(save_path)

    # return upstream base model
    if 'roberta' in model_config['name']:
        return model.roberta
    elif 'bert' in model_config['name']:
        return model.bert
    else:
        raise NotImplementedError


def trainTransfer(transfer_sequence: List[str], load_if_exists=True, **kwargs):
    if 'config' in kwargs:
        config = kwargs["config"]
    else:
        config = default_config()
        print("[trainTransfer]: Using default config.")
        print(json.dumps(config, indent=4))

    # Configurations
    model_config = config['model']
    trainer_config = config['trainer']

    # Run single task transfer
    prev_base_model = None
    cur_dir = []
    for task in transfer_sequence:
        cur_dir.append(task)
        prev_base_model = trainSingleTransfer(task, config, prev_base_model, cur_dir, load_if_exists)