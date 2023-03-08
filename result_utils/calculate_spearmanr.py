from scipy import stats
import json
from pprint import pprint

def getPercentageResult(raw_result):
	percentage_result = {}
	for k, v in raw_result.items():
		if len(k.split('-')) == 1:
			percentage_result.update({k: 0.0})
		else:
			base_performance = raw_result[k.split('-')[-1]]
			percentage_result.update({k: (v - base_performance) / base_performance * 100.0})
	return percentage_result

def getABC(raw_result):
	last_column = dict(raw_result)
	delete_keys = []
	for k in last_column:
		if len(k.split('-')) != 3:
			delete_keys.append(k)
	for k in delete_keys:
		del last_column[k]
	return last_column, [v for k, v in last_column.items()]

def getPartial(raw_result, keys, indices=[0, 2]):
	ret_list = []
	for k in keys:
		new_key = k.split('-')[indices[0]] + '-' + k.split('-')[indices[1]]
		ret_list.append(raw_result[new_key])
	return ret_list

if __name__ == '__main__':
	with open('../all_result.json', 'r') as f:
		raw_result = json.load(f)
	percentage_result = getPercentageResult(raw_result)
	#	percentage_result = raw_result
	dict_ABC, list_ABC = getABC(percentage_result)
	list_AB = getPartial(percentage_result, dict_ABC.keys(), indices=[0, 1])
	list_AC = getPartial(percentage_result, dict_ABC.keys(), indices=[0, 2])
	list_BC = getPartial(percentage_result, dict_ABC.keys(), indices=[1, 2])
	spearmanr_result = dict()
	sr = stats.spearmanr	
	spearmanr_result.update({'AB_ABC': sr(list_AB, list_ABC)})
	spearmanr_result.update({'AC_ABC': sr(list_AC, list_ABC)})
	spearmanr_result.update({'BC_ABC': sr(list_BC, list_ABC)})
	print(json.dumps(spearmanr_result, indent=4))
