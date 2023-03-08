import yaml, os, json
import torch

import pandas as pd

from itertools import permutations
from transformers import RobertaConfig, RobertaAdapterModel, BertConfig, BertAdapterModel
from typing import List, Union
from datasets import load_dataset
from transformers import TrainingArguments, AdapterTrainer
from tqdm import tqdm
from collections import defaultdict

from data_utils.DataManager import DatasetManager

file_name_mapping = {
    "rte": "RTE.tsv",
    "mrpc": "MRPC.tsv",
    "stsb": "STS-B.tsv",
    "cola": "CoLA.tsv",
    "qqp": "QQP.tsv",
    "sst2": "SST-2.tsv",
    "mnli": "MNLI-m.tsv",
    "qnli": "QNLI.tsv"
}

device = 'cuda'

def testSequence(sequence: List[str], model: Union[BertAdapterModel, RobertaAdapterModel]):
    tasks = []
    for task in sequence:
        tasks.append(task)
        root_folder = f'output_adapter1/{os.path.join(*tasks)}'

        # create output folder
        out_dir = f'test_result_qpoi/{root_folder}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, file_name_mapping[task])

        if os.path.exists(out_path):    # already predicted before
            print(f'[testSequence]: {out_path} exists, continue.')
            continue
        
        print(f'[testSequence]: processing {out_path}')
        
        adapter_name = model.load_adapter(f'{root_folder}/checkpoint-best/adapter', load_as=task)
        head_name = model.load_head(f'{root_folder}/checkpoint-best/head', load_as=adapter_name)
        model.set_active_adapters(adapter_name)     # 這行會噴錯... 很怪...
        model.to(device)

        data_manager = DatasetManager(task)
        test_set = data_manager.getDataSplit('test')

        model.eval()
        results = defaultdict(list)
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_set)):
                output = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device)
                )
                logits = output.logits

                results['id'].append(i)

                if task == "stsb":
                    preds = float(torch.squeeze(logits))
                    results['label'].append(preds)
                else:
                    preds = torch.argmax(logits, dim=1)
                    results['label'].append(int(preds))
                
        # store result as tsv
        df = pd.DataFrame(results)
        df.to_csv(out_path, sep='\t', index=False)

        # remove adapter & head from the model
        model.set_active_adapters(None)
        model.delete_adapter(adapter_name)
        model.delete_head(adapter_name)
    return

def main():
    with open('config.yaml', 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)
    
    tasks = config['exp_setup']['tasks']['name']

    # load pretrained model
    model_config = config['model']
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

    for sequence in list(permutations(tasks, config['exp_setup']['chain_length']))[0:6]:
        print(f"=*=*=*= Sequence: {sequence} =*=*=*=")
        testSequence(sequence, model)

if __name__ == '__main__':
    main()

