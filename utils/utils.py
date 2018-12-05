import scipy.io
import os
import torch
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
import math
from six.moves import xrange
from opt import opt
from torch.autograd import Variable
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
def clip_images_to_tensor(video_imgs, num_frame_per_clip=16, crop_size=112):
	data = []
	tmp_data = video_imgs
	img_datas = []
	if (len(tmp_data)!=0):
		for j in xrange(len(tmp_data)):
			img = Image.fromarray(tmp_data[j].astype(np.uint8))
			if img.width > img.height:
				scale = float(crop_size)/float(img.height)
				img = np.array(cv2.resize(np.array(img),(int(img.width*scale + 1), crop_size))).astype(np.float32)#以短边为准同比例缩放
			else:
				scale = float(crop_size)/float(img.width)
				img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height*scale +1)))).astype(np.float32)
			crop_x = int((img.shape[0] - crop_size)/2)
			crop_y = int((img.shape[1] - crop_size)/2)
			img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:]
			img_datas.append(img)
	data.append(img_datas)
	img_array = np.array(data).astype(np.float32)
	img_array = img_array.transpose((0,1,4,2,3))
	return img_array


def get_rand_frames(video_frame_numbers, output_frame_numbers=32):
	rand_frames = np.zeros(output_frame_numbers)
	div = float(video_frame_numbers) / float(output_frame_numbers)
	scale = math.floor(div)
	if scale == 0:
		rand_frames[0:video_frame_numbers] = np.arange(0, video_frame_numbers)
		rand_frames[video_frame_numbers::] = video_frame_numbers - 1
	elif scale == 1:
		rand_frames[0:output_frame_numbers] = div * np.arange(0, output_frame_numbers)
	else:
		rand_frames[::] = div*np.arange(0, output_frame_numbers) + float(scale)/2 * (np.random.random(size=output_frame_numbers)-0.5)
		rand_frames[0] = max(rand_frames[0], 0)
		rand_frames[output_frame_numbers - 1] = min(rand_frames[output_frame_numbers - 1], video_frame_numbers-1)
		rand_frames = np.floor(rand_frames) + 1
	return rand_frames


def model_test(video, model):
	model.to(device)
	if video.shape[0] !=1:
		video = np.array(video).transpose((0,3,1,2))
		video = Variable(torch.from_numpy(video).float()).to(device)
		video = video.unsqueeze(0)
	else:
		video = torch.from_numpy(video).float().to(device)
	#model.eval()
	out, out_logits = model(video)
	score, predicted_label = torch.max(out, 1)
	return predicted_label, score





