import time
from datetime import datetime

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
