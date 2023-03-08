import os
import json
import wandb

from typing import List, Union
from transformers import RobertaConfig, RobertaAdapterModel, BertConfig, BertAdapterModel
from transformers import TrainingArguments, AdapterTrainer, EarlyStoppingCallback

from data_utils.DataManager import DatasetManager

def default_config():
    return {
        "trainer": {
            "train_step": 6000,
            "logging_step": 250,
            "eval_step": 250,
            "save_step": 250,
            "save_total_limit": 1,
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
    model: Union[BertAdapterModel, RobertaAdapterModel],
    output_root: str,
    curr_chain: list,
    load_if_exists=True
):
    trainer_config = config['trainer']
    model_config = config['model']
    task_config = config['exp_setup']['tasks']

    curr_dir = os.path.join(output_root, *curr_chain)

    if load_if_exists == True and os.path.exists(f'{curr_dir}/{trainer_config["ckpt_path"]}'):
        model.load_adapter(f'{curr_dir}/{trainer_config["ckpt_path"]}/adapter')
        print(f'{curr_dir}/{trainer_config["ckpt_path"]} found. Weights are loaded and skip training.')
        return

    # using wandb for logging messgae
    wandb.init(project="seed_1209", group=task, name="-".join(curr_chain))
    wandb.config.update(config)

    print(f'{curr_dir}/{trainer_config["ckpt_path"]} not found. Start training...')

    # Start Training
    model.train_adapter('adapter')        # 先這樣寫
    data_manager = DatasetManager(task, tokenizer=model_config['name'], data_seed=trainer_config['seed'], size=trainer_config['size'])
    model.add_classification_head(task, num_labels=data_manager.getNumLabels())    # 如果要用不同類型的，這邊就寫個 if

    # Arguments for AdapterTrainer
    training_args = TrainingArguments(
        learning_rate=float(task_config['lr'][task]),
        per_device_train_batch_size=trainer_config['batch_size'],
        max_steps=trainer_config['train_step'],
        save_steps=trainer_config['save_step'],
        save_total_limit=trainer_config['save_total_limit'],
        eval_steps=trainer_config['eval_step'],
        evaluation_strategy='steps',
        seed=trainer_config['seed'],

        logging_steps=trainer_config['logging_step'],
        logging_dir=trainer_config['logging_dir'],

        load_best_model_at_end=trainer_config['load_best_model_at_end'],
        metric_for_best_model=trainer_config['metric_for_best_model'] + task_config['metrics'][task],

        output_dir=f"{curr_dir}",
        overwrite_output_dir=trainer_config['overwrite_output_dir'],
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=trainer_config['remove_unused_columns'],
        report_to="wandb"
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=data_manager.getDataSplit('train'),
        eval_dataset=data_manager.getDataSplit('eval'),
        compute_metrics=data_manager.getMetric(),
#        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )
    
    trainer.train()
    res = trainer.evaluate()
    with open(f'{curr_dir}/result.json', 'w') as f:
        json.dump(res, f, indent=4)
    
    os.mkdir(f'{curr_dir}/{trainer_config["ckpt_path"]}')

    # Save result
    model.save_adapter(f'{curr_dir}/{trainer_config["ckpt_path"]}/adapter', 'adapter')
    model.save_head(f'{curr_dir}/{trainer_config["ckpt_path"]}/head', task)
    model.delete_head(task)

    wandb.finish()


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

    # Run single task transfer
    output_root = f'seed_{trainer_config["seed"]}'
    curr_chain = []
    model.add_adapter('adapter', config='pfeiffer')
    model.train_adapter('adapter')     # 這行會 freeze model weight
    for task in transfer_sequence:
        curr_chain.append(task)
        trainSingleTransfer(task, config, model, output_root, curr_chain, load_if_exists)
