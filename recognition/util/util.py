"""
utils module 
"""
import time
from datetime import datetime

#optim
class AverageMeter(object):
    '''Compute and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value*n
        self.count += n
        self.average = self.sum/self.count


def calculate_accuracy(outputs, targets):
     '''
     args:
     outputs:output of model,shape of(N,C)
     targets:labels,shape of (N,1)
     '''
     batch_size = targets.size(0)
     _, pred = outputs.topk(1, 1, True) #shape(N,1)  
     pred= pred.t() #shape (1,N)
     correct = pred.eq(targets.view(1, -1).long())  #targets.view(1,-1) have not changed the shape of targets
     corrects_accu = correct.float().sum().data[0]
     return  corrects_accu / batch_size


#time
def print_time(work,start,end):
	hour = end.tm_hour - start.tm_hour
	minu = end.tm_min - start.tm_min
	sec = end.tm_sec - start.tm_sec
	if hour<0:
		hour += 24
	if minu<0:
		minu += 60
		hour -= 1
	if sec<0:
		sec += 60
		minu -= 1
	print('%s cost time:%dh%dm%ds'%(work, hour, minu, sec))


def get_strtime():
	time = '%s'%datetime.now()
	time1 = time.split(' ')[0]
	time2 = time.split(' ')[1].split('.')[0]
	strtime = '%s %s'%(time1, time2)
	return strtime
