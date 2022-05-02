import yaml, os, json
from itertools import permutations
from transformers import RobertaConfig, RobertaAdapterModel, BertConfig, BertAdapterModel
from typing import List, Union

def testSequence(sequence: List[str], model: Union[BertAdapterModel, RobertaAdapterModel]):
    cur_dir = []
    for task in sequence:
        cur_dir.append(task)
        if os.path.exists(f'{os.path.join(*cur_dir)}/test_result.csv'):    # already predicted before
            continue
        
        # load best checkpoint
        with open(f'{os.path.join(*cur_dir)}/checkpoint-6000/trainer_state.json') as f:
            tmp_json = json.load(f)
            best_ckpt = os.path.basename(tmp_json['best_model_checkpoint'])
        adapter_name = model.load_adapter(f'{os.path.join(*cur_dir)}/{best_ckpt}/adapter')
        model.set_active_adapters(adapter_name)     # 這行會噴錯... 很怪...
        model.load_head(f'{os.path.join(*cur_dir)}/{best_ckpt}/{task}')
        
        # predict and output in [cur_dir/test_result.csv]

        # delete head
        model.delete_head(task)
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
        testSequence(sequence, model)


if __name__ == '__main__':
    main()