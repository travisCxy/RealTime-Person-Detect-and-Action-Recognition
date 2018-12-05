import torch

def write_log(log_file, list1, list2, num_batch):
	log_file.write('Batch'+str(num_batch)+' predicted:')
	for i,val in enumerate(list1):
		log_file.write(str(int(val))+' ')
	log_file.write('\n'+'Batch'+str(num_batch)+' labels   :')
	for i ,val in enumerate(list2):
		log_file.write(str(int(val))+' ')
	log_file.write('\n')

#log_file = open('log_file.txt', 'w+')
#a = torch.Tensor([1,2,3,4])
#b = torch.Tensor([1,23,3,2])
#write_log(log_file,a,b,1)
