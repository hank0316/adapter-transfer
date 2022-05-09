import json
import os

TASKS = ['cola', 'rte', 'mrpc', 'stsb']
FINAL_CHECKPOINT = 2000

def traverse(cur_prefix, result):
	path_prefix = '/'.join(cur_prefix)
	if os.path.exists(f'./{path_prefix}/checkpoint-{FINAL_CHECKPOINT}'):
		with open(f'./{path_prefix}/checkpoint-{FINAL_CHECKPOINT}/trainer_state.json') as f:
			trainer_state = json.load(f)
		key = '-'.join(cur_prefix)
		value = trainer_state['best_metric']
		result.update({key : value})
	for task in TASKS:
		if os.path.exists(f'./{path_prefix}/{task}'):
			cur_prefix.append(task)
			traverse(cur_prefix, result)
			cur_prefix.pop()

if __name__ == '__main__':
	result = {}
	traverse([], result)
	with open('all_result.json', 'w') as f:
		json.dump(result, f, indent=4)
