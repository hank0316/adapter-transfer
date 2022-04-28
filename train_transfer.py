import os
import json

from typing import List, Union
from transformers import RobertaConfig, RobertaModelWithHeads, BertConfig, BertModelWithHeads
from transformers import TrainingArguments, AdapterTrainer

from data_utils.DataManager import DatasetManager

def default_config():
    return {
        "trainer": {
            "train_step": 6000,
            "logging_step": 50,
            "eval_step": 50,
            "save_step": 50,
            "load_best_model_at_end": True,
            "overwrite_output_dir": True,
            "remove_unused_columns": False,
            "ckpt_path": "checkpoint-best"
        },
        "model": {
            "name": "roberta-base"
        }
    }


def trainSingleTransfer(
    task: str, 
    trainer_config: dict, 
    model: Union[BertModelWithHeads, RobertaModelWithHeads],
    cur_dir: list,
    load_if_exists=True
):

    if load_if_exists == True and os.path.exists(f'{os.path.join(*cur_dir)}/{trainer_config["ckpt_path"]}'):
        model.load_adapter(f'{os.path.join(*cur_dir)}/{trainer_config["ckpt_path"]}/adapter')
        print(f'{os.path.join(*cur_dir)}/{trainer_config["ckpt_path"]} found. Weights are loaded and skip training.')
        return

    print(f'{os.path.join(*cur_dir)}/{trainer_config["ckpt_path"]} not found. Start training...')
    # Start Training
    model.train_adapter('adapter')        # 先這樣寫
    data_manager = DatasetManager(task)
    model.add_classification_head(task, num_labels=data_manager.getNumLabels())    # 如果要用不同類型的，這邊就寫個 if

    # Arguments for AdapterTrainer
    training_args = TrainingArguments(    # 我在想這些是不是也要按 task 去調整
        learning_rate=trainer_config['learning_rate'],
        max_steps=trainer_config['train_step'],
        logging_steps=trainer_config['logging_step'],
        save_steps=trainer_config['save_step'],
        eval_steps=trainer_config['eval_step'],
        evaluation_strategy='steps',
        logging_dir=trainer_config['logging_dir'],
        load_best_model_at_end=trainer_config['load_best_model_at_end'],
        output_dir=f"{os.path.join(*cur_dir)}",
        overwrite_output_dir=trainer_config['overwrite_output_dir'],
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=trainer_config['remove_unused_columns'],
    )

    trainer = AdapterTrainer(
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
    
    os.mkdir(f'{os.path.join(*cur_dir)}/{trainer_config["ckpt_path"]}')

    # Save result
    model.save_adapter(f'{os.path.join(*cur_dir)}/{trainer_config["ckpt_path"]}/adapter', 'adapter')
    model.save_head(f'{os.path.join(*cur_dir)}/{trainer_config["ckpt_path"]}/head', task)
    model.delete_head(task)


def trainTransfer(transfer_sequence: List[str], load_if_exists=True, **kwargs):
    if 'config' in kwargs:
        config = kwargs["config"]
    else:
        config = default_config()
        print("[trainTransfer]: Using default config.")
        print(json.dumps(config, indent=4))

    model_config = config['model']
    trainer_config = config['trainer']
    # Load Model
    if 'roberta' in model_config['name']:
        pretrain_config = RobertaConfig.from_pretrained(model_config["name"])
        model = RobertaModelWithHeads.from_pretrained(
            model_config["name"], 
            config=pretrain_config
        )
    elif 'bert' in model_config['name']:
        pretrain_config = BertConfig.from_pretrained(model_config["name"])
        model = BertModelWithHeads.from_pretrained(
            model_config["name"], 
            config=pretrain_config
        )
    else:
        raise

    cur_dir = []
    model.add_adapter('adapter')
    model.train_adapter('adapter')     # 這行會 freeze model weight
    for task in transfer_sequence:
        cur_dir.append(task)
        trainSingleTransfer(task, trainer_config, model, cur_dir, load_if_exists)