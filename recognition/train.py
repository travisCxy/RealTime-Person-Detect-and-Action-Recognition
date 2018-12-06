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
from data.ISO_GD.dataloader import get_dataloaders
from data.ntu_dataset.dataloader import get_ntu_dataloaders
from data.UCF101.dataloader import *
from net.load_model import load_model, weight_init, show_model, fix_layer
from util.print_time import print_time, get_strtime

torch.backends.cudnn.benchmark = True
args = opt


args.listpath = './data/ntu_dataset/list/cross_view'
args.datapath = '../../dataset/NTU/ntu_frame'
args.num_classes = 60
dataloaders = get_ntu_dataloaders(args)
print('Data Loaded...')
model = load_model(args)
params = filter(lambda p: p.requires_grad, model.parameters())
if args.model_path is not None:
	model_dict = model.state_dict()
	pretrained_model = torch.load(args.model_path)
	pretrained_model = {k:v for k,v in pretrained_model.items() if k in model_dict}
	model_dict.update(pretrained_model)
	model.load_state_dict(model_dict)
else:
	model.apply(weight_init)
if args.use_cuda:
	model.cuda()
if args.data_type == 'half':
	model.half()
print('Model Loaded...')
optimizer = optim.SGD(params, lr=args.LR, momentum=args.momentum, weight_decay=args.weight_decay)
epoch = 1

while epoch<args.num_epochs:
	
	valid_loss, valid_corrects, valid_steps = 0.0, 0.0, 0
	train_loss, train_corrects, train_steps = 0.0, 0.0, 0
	print('Training...')
	if args.work == 'TV' or args.work == 'T':
		for data in dataloaders['train']:
			inputs, labels = data['video'], data['label']
			if 'joint' in data:
				joint = data['joint']
				inputs = {'video':inputs, 'joint':joint}
			model.train()
			if bool(args.use_cuda):
				if 'joint' in data:
					inputs['video'] = Variable(inputs['video'].float().cuda())
					labels = Variable(labels.float().cuda())
				else:
					inputs = Variable(inputs.float().cuda())
					labels = Variable(labels.float().cuda())
			else:
				if 'joint' in data:
					inputs['video'] = Variable(inputs['video'].float())
					labels = Variable(labels.float())
				else:
					inputs = Variable(inputs.float())
					labels = Variable(labels.float())
			out, out_logits = model(inputs)
			_, predicted = torch.max(out, 1)
			optimizer.zero_grad()
			loss = F.cross_entropy(out_logits, labels.long().squeeze())
			corrects = ((predicted.short() == labels.short().squeeze()).sum())
			if args.data_type == 'half':
				loss.half()
				corrects.half()
			train_loss += loss.item()
			train_corrects += corrects.item()
			train_steps += labels.size(0)
			loss.backward()
			optimizer.step()
			if train_steps % args.print_step ==0:
				time = get_strtime()
				print('Training  '+str(time)+'epoch:'+str(epoch)+'  iterations:'+str(train_steps)+
				  '  loss:%3f'%(train_loss/train_steps)+'  accuracy:%3f'%(train_corrects/train_steps))
		time = '%s'%datetime.now()
		time1 = time.split(' ')[0]
		if not os.path.exists('model/iso_model/'+str(time1)):
			os.mkdir('model/iso_model/'+str(time1))
		torch.save(model.state_dict(), 'model/iso_model/'+str(time1)+'/'+str(args.note)+'_'+str(epoch))
		print('model has been saved at model/iso_model/'+str(time1)+'/'+str(args.note)+'_'+str(epoch))
	if args.work == 'TV' or args.work == 'V':
		print('Valding...')
		for data in dataloaders['valid']:
			inputs, labels = data['video'], data['label']
			model.eval()
			if 'joint' in data.keys():
				joint = data['joint']
				inputs = {'video':inputs, 'joint':joint}
			if bool(args.use_cuda):
				model.cuda()
				if  isinstance(inputs, dict):
					inputs['video'] = Variable(inputs['video'].float().cuda())
					labels = Variable(labels.cuda())
				else:
					inputs = Variable(inputs.float().cuda())
					labels = Variable(labels.float().cuda())
			else:
				if isintance(inputs, dict):
					inputs['video'] = Variable(inputs['video'].float())
					labels = Variable(labels.float())
				else:
					inputs = Variable(inputs.float())			
					labels = Variable(labels.float())
			out, out_logits = model(inputs)
			_, predicted = torch.max(out, 1)
			#print()
			loss = F.cross_entropy(out_logits, labels.long().squeeze())
			corrects = ((predicted.short() == labels.short().squeeze()).sum())
			if args.data_type == 'half':
				loss.half()
				corrects.half()
			valid_loss += loss.item()
			valid_steps += labels.size(0)
			valid_corrects += corrects.item()
			if valid_steps % args.print_step ==0:
				print('validing  '+'step:'+str(valid_steps)+'  accuracy:'+str(valid_corrects/valid_steps))
	
	epoch += 1

	
