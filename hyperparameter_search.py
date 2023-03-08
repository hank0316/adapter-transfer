import yaml
import wandb
from itertools import permutations

from typing import List, Union
from transformers import RobertaConfig, RobertaAdapterModel, BertConfig, BertAdapterModel
from transformers import TrainingArguments, AdapterTrainer, EarlyStoppingCallback

from data_utils.DataManager import DatasetManager

import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp2/b08902126/cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp2/b08902126/cache"
os.environ["WANDB_PROJECT"] = "adapter_hyperparameter_search"
os.environ["WANDB_ENTITY"] = "miulab_transfer_learning"

def main():
    with open('config.yaml', 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)
    
    # Configurations
    model_config = config['model']
    trainer_config = config['trainer']
    task_config = config['exp_setup']['tasks']

    # Hyperparameter Search
    f = open('./parameter_search/log.txt', 'w')
    tasks = config['exp_setup']['tasks']['name']
    for task in tasks:
        print(f'===== Searching Hyperparameter for {task} =====')

        os.environ["WANDB_PROJECT"] = f'adapter_hyperparameter_search_{task}'

        data_manager = DatasetManager(task, tokenizer=model_config['name'], data_seed=trainer_config['seed'], size=trainer_config['size'])

        def model_init(trial):
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
            
            model.add_adapter('adapter', config='pfeiffer')
            model.train_adapter('adapter')
            model.add_classification_head(task, num_labels=data_manager.getNumLabels())

            return model

        def wandb_hp_space(trial):
            return {
                "method": "grid",
                "metric": {
                    "name": "objective", 
                    "goal": "minimize"
                },
                "parameters": {
                    "learning_rate": {
                        # Is there any better way to generate such list?
                        # "values": [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4]
                        "values": [2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]
                    }
                },
            }

        # Arguments for AdapterTrainer
        training_args = TrainingArguments( 
            learning_rate=trainer_config['learning_rate'],
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

            output_dir=f"parameter_search/{task}/",
            overwrite_output_dir=trainer_config['overwrite_output_dir'],
            # The next line is important to ensure the dataset labels are properly passed to the model
            remove_unused_columns=trainer_config['remove_unused_columns'],
        )

        model = model_init(None)

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=data_manager.getDataSplit('train'),
            eval_dataset=data_manager.getDataSplit('eval'),
            compute_metrics=data_manager.getMetric(),
            model_init=model_init
        )

        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="wandb",
            hp_space=wandb_hp_space,
            n_trials=10,
        )

        print(f'best trial of task {task}: {best_trial}', file=f)

if __name__ == '__main__':
    main()
