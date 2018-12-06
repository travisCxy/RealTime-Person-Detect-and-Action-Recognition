import os
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as Img
import math
import sys
sys.path.append('./data')

from torchvision import transforms
from torch.utils.data import Dataset,DataLoader 
from torchvision import transforms,utils
from scipy.misc import imread,imresize 
from PIL import Image
from transformer import *	

class NTU_RGB(Dataset):
	def __init__(self, info_list, box_info_path, root_dir,transform=None, output_frame_numbers=32):
		'''
		Argument
		info_list:The list include path/to/data and label
		root_dir: The directory of data
		transform: Transformer for sample
		output_frame_numbers: frame_num of output image sequence
		'''
		self.information = pd.read_csv(info_list, delimiter=' ', header=None)
		self.root_dir = root_dir
		self.transform = transform
		self.output_frame_numbers = output_frame_numbers
		self.info_list = info_list
		self.box_info_path = box_info_path
		
	def __len__(self):
		return len(self.information)

	def __getitem__(self, idx):
		video_path = os.path.join(self.root_dir, self.information.iloc[idx, 0])
		box_path = os.path.join(self.box_info_path, self.information.iloc[idx, 0])
		box_infp = read_box_info(box_path)
		video_frame_numbers = self.information.iloc[idx, 1]
		label = self.information.iloc[idx, 2]
		video,rand_frames = self.get_single_video(video_path, video_frame_numbers)
		sample = {'video':video, 'label':label, 'rand_frames':rand_frames, 'box_info':box_info}
		if self.transform:
			sample = self.transform(sample)
		return sample

	def get_single_video(self, video_path, video_frame_numbers):
		output_frame_numbers = self.output_frame_numbers
		assert os.path.exists(video_path)
		rand_frames = np.zeros(output_frame_numbers)
		div = float(video_frame_numbers) / float(output_frame_numbers)
		scale = math.floor(div)
		
		if scale == 0:
			rand_frames[0: video_frame_numbers] = np.arange(0, video_frame_numbers)
			rand_frames[video_frame_numbers::] = video_frame_numbers - np.arange(2,output_frame_numbers-video_frame_numbers+2)
		elif scale ==1:
			rand_frames[0:output_frame_numbers] = div*np.arange(0, output_frame_numbers)
		else:
			rand_frames[::] = div*np.arange(0, output_frame_numbers) + float(scale)/2*(np.random.random(size=output_frame_numbers) - 0.5)
		rand_frames[0] = max(rand_frames[0], 0)
		rand_frames[output_frame_numbers - 1] = min(rand_frames[output_frame_numbers-1], video_frame_numbers-1)
		rand_frames = np.floor(rand_frames) + 1
		
		processed_images = np.empty((output_frame_numbers, 1080, 1920, 3), dtype=np.float32)
		for idx in range(0, output_frame_numbers):
			image_file = '%s/%04d.jpg'%(video_path, rand_frames[idx])
			assert os.path.exists(image_file)
			processed_images[idx] = imread(image_file)	#(h,w,3) rgb
		return processed_images, rand_frames
	
transformer = transforms.Compose([crop_by_box(), ToTensor()])
train_list = os.path.join(args.list_path, 'train_list.txt')
train_dataset = NTU_RGB(train_list, opt.box_info_path, root_dir=opt.data_path, transform=transformer)
train_dataloader = DataLoader(train_dataset, batch_size=args.trainBatch, shuffle=True, num_workers=args.num_workers)
valid_list = os.path.join(args.list_path, 'valid_list.txt')
valid_dataset = NTU_RGB(valid_list, opt.box_info_path, root_dir=opt.data_path, transform=transformer)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.validBatch, shuffle=True, num_workers=args.num_workers)
dataloaders = {'train':train_dataloader, 'valid':valid_dataloader}
