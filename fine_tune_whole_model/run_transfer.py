import yaml
from itertools import permutations

from train_transfer import trainTransfer

import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp2/b08902126/cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp2/b08902126/cache"

def main():
    with open('config.yaml', 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)
    
    tasks = config['exp_setup']['tasks']['name']
    for sequence in permutations(tasks, config['exp_setup']['chain_length']):
        print('=*=*=*=*= Current Sequence =*=*=*=*=')
        print(list(sequence))
        print('=*=*=*=*= Current Sequence =*=*=*=*=')
        trainTransfer(list(sequence), load_if_exists=True, config=config)

if __name__ == '__main__':
    main()