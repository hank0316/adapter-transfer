import yaml
from itertools import permutations, combinations

from train_mtl import trainMTL

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
    for subset in combinations(tasks, config['exp_setup']['num_multi_task']):
        print('=*=*=*=*= Current Task Set =*=*=*=*=')
        print(list(subset))
        print('=*=*=*=*= Current Task Set =*=*=*=*=')
        trainMTL(list(subset), load_if_exists=True, config=config)

if __name__ == '__main__':
    main()
