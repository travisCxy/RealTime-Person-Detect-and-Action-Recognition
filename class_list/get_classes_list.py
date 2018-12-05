import pandas as pd


def get_classes_list(list_path, dataset):
	list = open(list_path, 'r')
	classes = []
	for line in list:
		if dataset == 'UCF':
			note = line.split(' ')[-1].strip()
		elif dataset == 'NTU':
			note = line.split('. ')[-1].strip()
		classes.append(note) 
	return classes
'''
list_path = 'NTUClasses.txt'
classes = get_classes_list(list_path, 'NTU')
print((classes))	
'''
