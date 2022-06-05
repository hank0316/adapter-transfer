import torch

from data_utils.DataManager import DatasetManager

class MTL_Manager:
	def __init__(self, tasks: list, tokenizer='roberta-base', size=1000, data_seed=316):
		self.tasks = tasks
		self.managers = {task : DatasetManager(task, tokenizer, size, data_seed) for task in self.tasks}
	
	def getDataSplit(self, split='train'):
		return {task : self.managers[task].getDataSplit(split) for task in self.tasks}

	def getMetric(self):
		return {task : self.managers[task].getMetric() for task in self.tasks}

	def getCriteria(self):
		pass

	def getNumLabels(self):
		return {task : self.managers[task].getNumLabels() for task in self.tasks}


