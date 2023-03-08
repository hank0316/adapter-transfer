import json
import yaml
import re
import numpy as np
import pandas as pd
import sys
from scipy import stats
from pathlib import Path
from collections import OrderedDict, defaultdict


class ResultViewer:
    def __init__(self, result_path, omit_tasks=[]):
        result_path = Path(result_path)
        if result_path.suffix == '.json':
            with open(result_path, 'r') as f:
                print(f'[ResultViewer] loading results from "{result_path}"', file=sys.stderr)
                self.results = json.load(f)

        self.results = self._omit_tasks(self.results, omit_tasks)
        self.perf_gain = self._get_perf_gain(self.results)

        self.start_with = self.source
        self.end_with = self.target

    def source(self, task, chain_length=-1, output_perf_gain=False):
        return self._find_matching_results(f'^{task}', chain_length, output_perf_gain)

    def target(self, task, chain_length=-1, output_perf_gain=False):
        return self._find_matching_results(f'.*{task}$', chain_length, output_perf_gain)

    def transferability(self, task1, task2, two_way=False):  # if two_way == False, return task1 -> task2
        perf_gain_1 = self.perf_gain[f'{task2}-{task1}']
        perf_gain_2 = self.perf_gain[f'{task1}-{task2}']
        return (perf_gain_1 + perf_gain_2) / 2. if two_way else perf_gain_2

    def spearmanr(self, two_way=False):
        AB, AC, BC, ABC = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        long_chain = self._find_matching_results(pattern='.*', chain_length=3)  
        for task_chain, target_performance in long_chain.items():
            A, B, C = task_chain.split('-')
            AB[C].append(self.transferability(A, B, two_way))
            AC[C].append(self.transferability(A, C, two_way))
            BC[C].append(self.transferability(B, C, two_way))
            ABC[C].append(self.perf_gain[task_chain])  # performance gain of C task in ABC chain
        sr = stats.spearmanr
        return {task: {'AB_ABC': sr(AB[task], ABC[task]), 'AC_ABC': sr(AC[task], ABC[task]), 'BC_ABC': sr(BC[task], ABC[task])} for task in AB.keys()}

    def spearmanr4(self, two_way=False):
        AD, BD, CD, ABCD = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        long_chain = self._find_matching_results(pattern='.*', chain_length=4)  
        for task_chain, target_performance in long_chain.items():
            A, B, C, D = task_chain.split('-')
            AD[D].append(self.transferability(A, D, two_way))
            BD[D].append(self.transferability(B, D, two_way))
            CD[D].append(self.transferability(C, D, two_way))
            ABCD[D].append(self.perf_gain[task_chain])  # performance gain of D task in ABCD chain
        sr = stats.pearsonr
        return {task: {'AD': AD[task], 'BD': BD[task], 'CD': CD[task], 'ABCD': ABCD[task]} for task in AD.keys()}
        # return {task: {'AD_ABCD': sr(AD[task], ABCD[task]), 'BD_ABCD': sr(BD[task], ABCD[task]), 'CD_ABCD': sr(CD[task], ABCD[task])} for task in AD.keys()}
        
    def _find_matching_results(self, pattern, chain_length=-1, output_perf_gain=False):
        working_dict = self.perf_gain if output_perf_gain else self.results
        keys = sorted([key for key in working_dict.keys() if re.match(pattern, key)])
        keys = sorted(keys, key=lambda x: x.count('-'))
        keys = self._filter_keys_by_chain_length(keys, chain_length)
        return ResultDict([(k, working_dict[k]) for k in keys])
        
    def _filter_keys_by_chain_length(self, keys, chain_length):
        if chain_length == -1:
            return keys
        return [key for key in keys if key.count('-') == chain_length - 1]

    def _get_perf_gain(self, results):
        perf_gain = dict()
        for k, v in results.items():
            tasks = k.split('-')
            perf_gain[k] = (v - results[tasks[-1]]) / results[tasks[-1]] * 100.0
        return perf_gain

    def _omit_tasks(self, tasks, to_be_omitted):
        omitted_result = {}
        for k in tasks.keys():
            need_insert = True
            for task in k.split('-'):
                if task in to_be_omitted:
                    need_insert = False
                    break
            if need_insert:
                omitted_result[k] = tasks[k]
        return omitted_result

    
class ResultDict(OrderedDict):
    @property
    def best(self):
        key = max(self, key=self.get)
        return {key: self[key]}
    
    @property
    def mean(self):
        return np.mean(list(self.values()))

    @property
    def std(self):
        return np.std(list(self.values()))
