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
    "cola": "CoLA.tsv"
}
device = 'cuda:1'

def testSequence(sequence: List[str], model: Union[BertAdapterModel, RobertaAdapterModel]):
    cur_dir = []
    for task in sequence:
        cur_dir.append(task)
        # if os.path.exists(f'{os.path.join(*cur_dir)}/{file_name_mapping[task]}'):    # already predicted before
        #    continue
        
        # load best checkpoint
        with open(f'{os.path.join(*cur_dir)}/checkpoint-6000/trainer_state.json') as f:
            tmp_json = json.load(f)
            best_ckpt = os.path.basename(tmp_json['best_model_checkpoint'])

        adapter_name = model.load_adapter(f'{os.path.join(*cur_dir)}/{best_ckpt}/adapter')
        model.load_head(f'{os.path.join(*cur_dir)}/{best_ckpt}/{task}', load_as=adapter_name)
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
                if task == "stsb":
                   preds = torch.squeeze(logits)
                else:
                   preds = torch.argmax(logits, dim=1)
                results['id'].append(i)
                results['label'].append(int(preds))
#        with open(f'{os.path.join(*cur_dir)}/test_result.json', 'w') as f:
#            json.dump(res, f, indent=4)
        df = pd.DataFrame(results)
        df.to_csv(f'{os.path.join(*cur_dir)}/{file_name_mapping[task]}', sep='\t', index=False)
        # predict and output in [cur_dir/test_result.csv]
	
        # delete head
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

    for sequence in permutations(tasks, config['exp_setup']['chain_length']):
        print(f"=*=*=*= Sequence: {sequence} =*=*=*=")
        testSequence(sequence, model)


if __name__ == '__main__':
    main()
