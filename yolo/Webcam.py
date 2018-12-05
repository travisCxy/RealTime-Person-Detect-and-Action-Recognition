import argparse
import time
import torch
import cv2
import queue
import sys
import os
import math
import numpy as np
import pyrealsense2 as rs

from models import *
from yolo_utils.datasets import *
from yolo_utils.utils import *
from threading import Thread
from detect_opt import opt


class camera_detect:
	def __init__(self):
		#self.Q = queue.LifoQueue()
		self.stopped = False
		self.model = Darknet(opt.cfg, opt.img_size)
		weights_path = opt.detect_weights_path
		if weights_path.endswith('.weights'):
			load_weights(self.model, weights_path)
		else:
						
			checkpoint = torch.load(weights_path, map_location='cpu')
			self.model.load_state_dict(checkpoint['model'])
			del checkpoint
		self.model.to(device).eval()

	def start(self):
		t1 = Thread(target=self.read_detect, args=())
		t1.start()
		return self
	
	def read_detect(self):
		global detect_flag
		global detections
		global classes
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
		pipeline.start(config)
		try:
			while True:
				read_start = time.time()
				frames = pipeline.wait_for_frames()
				frames.keep()
				color_frame = frames.get_color_frame()
				if not color_frame:
					continue    # pass current loop 
				orig_image = np.asanyarray(color_frame.get_data())
				detect_size = opt.img_size
				img, _, _, _ = resize_square(orig_image, height=detect_size, color=(127.5, 127.5, 127.5))
				img = img[:, :, ::-1].transpose(2,0,1)
				img = np.ascontiguousarray(img, dtype=np.float32)
				img /= 255.0
				img = torch.from_numpy(img).unsqueeze(0).to(device)
				detect_start = time.time()
				predictions = self.model(img)
				predictions = predictions[predictions[:, :, 4]>opt.conf_thres]
				if len(predictions)>0:
					detections = non_max_suppression(predictions.unsqueeze(0), opt.conf_thres, opt.nms_thres)
				#print(detections)
					if detections is not None:
						detect_flag = True
				else:
					detect_flag = False 
				detect_end = time.time()
				if detect_flag:
					#print('detecting')
					orig_image, _ = draw_boxes(orig_image, detect_size, detections, classes)
				if opt.show:
					cv2.imshow('detect', orig_image)
					cv2.waitKey(1)
				print('read a frame cost:%.3f'%(detect_start-read_start))
				print('detect a frame cost:%.3f'%(detect_end - detect_start))
		except Exception as e:
			print(e)
		pass
	
	#def getitem(self):
		#return self.Q.get()
	
	def stop(self):
		self.stop=False
		
class cameraloader:
	def __init__(self):
		self.Q = queue.LifoQueue()
		self.stopped = False
		self.model = Darknet(opt.cfg, opt.img_size)
		weights_path = opt.detect_weights_path
		if weights_path.endswith('.weights'):
			load_weights(self.model, weights_path)
		else:
						
			checkpoint = torch.load(weights_path, map_location='cpu')
			self.model.load_state_dict(checkpoint['model'])
			del checkpoint
		self.model.to(device).eval()

	def start(self):
		t1 = Thread(target=self.read_detect, args=())
		t1.start()
		return self
	
	def read_detect(self):
		global detect_flag
		global detections
		global classes
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
		pipeline.start(config)
		try:
			while True:
				read_start = time.time()
				frames = pipeline.wait_for_frames()
				frames.keep()
				color_frame = frames.get_color_frame()
				if not color_frame:
					continue    # pass current loop 
				orig_image = np.asanyarray(color_frame.get_data())
				detect_size = opt.img_size
				img, _, _, _ = resize_square(orig_image, height=detect_size, color=(127.5, 127.5, 127.5))
				img = img[:, :, ::-1].transpose(2,0,1)
				img = np.ascontiguousarray(img, dtype=np.float32)
				img /= 255.0
				self.Q.put((img, orig_image))	
				read_end = time.time()
				#print('read a frame cost:%.3f'%(read_end-read_start))		
				if detect_flag:
					#print('detecting')
					orig_image, _ = draw_boxes(orig_image, detect_size, detections, classes)
				if opt.show:
					cv2.imshow('detect', orig_image)
					cv2.waitKey(1)
				
		except Exception as e:
			print(e)
		pass
	
	def getitem(self):
		return self.Q.get()
	
	def stop(self):
		self.stop=False

class detect:
	def __init__(self, cameraloader):
		self.cameraloader = cameraloader
		self.model = Darknet(opt.cfg, opt.img_size)
		weights_path = opt.detect_weights_path
		if weights_path.endswith('.weights'):
			load_weights(self.model, weights_path)
		else:
						
			checkpoint = torch.load(weights_path, map_location='cpu')
			self.model.load_state_dict(checkpoint['model'])
			del checkpoint
		self.model.to(device).eval()
		
	def start(self):
		t2 = Thread(target=self.detecting, args=())
		t2.start()
		return self
	
	def detecting(self):
		global detect_flag
		global detections
		while True:
			detect_start = time.time()
			img, orig_image = self.cameraloader.getitem()
			img = torch.from_numpy(img).unsqueeze(0).to(device)
			predictions = self.model(img)
			predictions = predictions[predictions[:, :, 4]>opt.conf_thres]
			if len(predictions)>0:
				detections = non_max_suppression(predictions.unsqueeze(0), opt.conf_thres, opt.nms_thres)
				#print(detections)
				if detections is not None:
					detect_flag = True
			else:
				detect_flag = False
			detect_end = time.time()
			print('detect one frame cost:%.3f'%(detect_end - detect_start))
				
if __name__ == '__main__':
	cuda = torch.cuda.is_available()
	device = torch.device('cuda:0' if cuda else 'cpu')

	detect_flag = False
	detections = None
	classes = load_classes(opt.class_path)
	'''
	cameradetect = camera_detect().start()
	cameradetect.read_detect()
	'''
	cameraloader = cameraloader().start()
	detect = detect(cameraloader).start()
	
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
