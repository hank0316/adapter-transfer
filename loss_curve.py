import json
import yaml
import os

from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np

def plot_curve(sequence: list[str], metric: dict, output_dir: str, max_step: int):
    for i in range(len(sequence)):
        tasks_dir = '/'.join(sequence[: i + 1])
        if not os.path.exists(f'{output_dir}/{tasks_dir}/checkpoint-{max_step}/'):
            continue
        with open(f'{output_dir}/{tasks_dir}/checkpoint-{max_step}/trainer_state.json', 'r') as f:
            # log_history: list(dict)
            log = json.load(f)['log_history']

            steps, loss = [], []
            eval_loss, eval_metric = [], []
            for j in range(0, len(log), 2):
                loss.append(log[j]['loss'])
                eval_loss.append(log[j + 1]['eval_loss'])
                eval_metric.append(log[j + 1][f'eval_{metric[sequence[i]]}'])
                steps.append(log[j]['step'])

            # plot loss
            fig, ax1 = plt.subplots() 

            ax1.plot(steps, loss, label='train_loss', marker=".")
            ax1.plot(steps, eval_loss, label='eval_loss', marker='.')
            ax1.set_xlabel('step')
            ax1.set_ylabel('loss')
            ax1.set_title('-'.join(sequence[: i + 1]))
            # ax1.legend(loc=0, bbox_to_anchor=(0.25, 1.15), bbox_transform=ax1.transAxes)

            # plot eval metric
            ax2 = ax1.twinx()
            ax2.set_ylabel(f'eval_{metric[sequence[i]]}')
            ax2.plot(steps, eval_metric, 'g', label=metric[sequence[i]], marker=".")
            # ax2.legend(loc=0, bbox_to_anchor=(1, 1.1), bbox_transform=ax1.transAxes)
            fig.legend(loc='upper right', bbox_to_anchor=(1.2, 1.21), bbox_transform=ax1.transAxes)

            plt.savefig(f'{output_dir}/{"/".join(sequence[: i + 1])}/result.png', bbox_inches='tight', dpi=100)
            plt.close()

def main():
    with open('config.yaml', 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)
    
    # please modified here and the config['exp_setup']['tasks'] (name, chain length, etc)
    output_dir = 'output_1122'
    max_step = '5000'

    tasks = config['exp_setup']['tasks']['name']
    for sequence in list(permutations(tasks, config['exp_setup']['chain_length'])):
        plot_curve(sequence, config['exp_setup']['tasks']['metrics'], output_dir, max_step)

if __name__ == '__main__':
    main()
