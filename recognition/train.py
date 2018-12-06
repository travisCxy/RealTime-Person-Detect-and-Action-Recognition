import os
import sys
import time
import numpy as np
from datetime import datetime
sys.path.append('./')
#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from option import opt
from data.dataloader import dataloaders
from net.I3D import I3D
from util.print_time import print_time, get_strtime

torch.backends.cudnn.benchmark = True
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

dataloader = dataloaders
print('Data Loaded...')
model = I3D(num_classes=opt.num_classes, input_size=opt.input_size)
if args.model_path is not None:
	model_dict = model.state_dict()
	pretrained_model = torch.load(args.model_path)
	pretrained_model = {k:v for k,v in pretrained_model.items() if k in model_dict}
	model_dict.update(pretrained_model)
	model.load_state_dict(model_dict)
else:
	model.apply(weight_init)
model.to(device)
print('Model Loaded...')

optimizer = optim.SGD(model.parameters(), lr=args.LR, momentum=args.momentum, weight_decay=args.weight_decay)
epoch = 1

while epoch<args.num_epochs:
	
	valid_loss, valid_corrects, valid_steps = 0.0, 0.0, 0
	train_loss, train_corrects, train_steps = 0.0, 0.0, 0
	print('Training...')
	if args.work == 'TV' or args.work == 'T':
		for data in dataloaders['train']:
			inputs, labels = data['video'], data['label']
			model.train()
			inputs.to(device)
			labels.to(device)
			out, out_logits = model(inputs)
			_, predicted = torch.max(out, 1)
			optimizer.zero_grad()
			loss = F.cross_entropy(out_logits, labels.long().squeeze())
			corrects = ((predicted.short() == labels.short().squeeze()).sum())
			train_loss += loss.item()
			train_corrects += corrects.item()
			train_steps += labels.size(0)
			loss.backward()
			optimizer.step()
			if train_steps % args.print_step ==0:
				time = get_strtime()
				print('training '+str(time)+'epoch:'+str(epoch)+'  iterations:'+str(train_steps)+
				  '  loss:%3f'%(train_loss/train_steps)+'  accuracy:%3f'%(train_corrects/train_steps))
		time = '%s'%datetime.now()
		time1 = time.split(' ')[0]
		if not os.path.exists('model/ntu_model/'+str(time1)):
			os.mkdir('model/ntu_model/'+str(time1))
		torch.save(model.state_dict(), 'model/ntu_model/'+str(time1)+'/'+str(args.note)+'_'+str(epoch))
		print('model has been saved at model/ntu_model/'+str(time1)+'/'+str(args.note)+'_'+str(epoch))
	if args.work == 'TV' or args.work == 'V':
		print('Valding...')
		for data in dataloaders['valid']:
			inputs, labels = data['video'], data['label']
			model.eval()
			inputs.to(device)
			labels.to(device)
			out, out_logits = model(inputs)
			_, predicted = torch.max(out, 1)
			loss = F.cross_entropy(out_logits, labels.long().squeeze())
			corrects = ((predicted.short() == labels.short().squeeze()).sum())
			valid_loss += loss.item()
			valid_steps += labels.size(0)
			valid_corrects += corrects.item()
			if valid_steps % args.print_step ==0:
				print('validing  '+'step:'+str(valid_steps)+'  accuracy:'+str(valid_corrects/valid_steps))
	
	epoch += 1

	
