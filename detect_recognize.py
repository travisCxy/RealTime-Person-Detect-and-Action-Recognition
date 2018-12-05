import os
import sys
import cv2
import time
import math
import torch
import queue
import threading
import numpy as np
import pyrealsense2 as rs

sys.path.append('./yolo')
from PIL import Image, ImageDraw
from utils.utils import *
from utils.preprocess import *
from threading import Thread
from opt import opt
#yolo module
from yolo.models import *
from yolo.yolo_utils.datasets import *
from yolo.yolo_utils.utils import *

#recognition module
from recognition.net.I3D import I3D

class CameraLoader:
	def __init__(self):
		self.Q = queue.LifoQueue()
		self.stopped = False
	
	def start(self):
		t1 = Thread(target=self.read_camera, args=())
		t1.start()
		return self
	
	def read_camera(self):
		global detect_flag 
		global detections 
		global classes
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
		pipeline.start(config)
		try:
			while True:
				frames = pipeline.wait_for_frames()
				frames.keep()
				color_frame = frames.get_color_frame()
				if not color_frame:
					continue
				orig_image = np.asanyarray(color_frame.get_data())
				detect_size = opt.detect_size
				img, _, _, _ = resize_square(orig_image, height=detect_size, color=(127.5,127.5,127.5)) #img is of shape (416,416,3)
				img = img[:, :, ::-1].transpose(2,0,1)  #bgr->rgb
				img = np.ascontiguousarray(img, dtype=np.float32)
				img /= 255.0
				self.Q.put((img, orig_image))
		except Exception as e:
			print(e)
		pass
	
	def get_item(self):
		return self.Q.get()
	
	def stop(self):
		self.stopped = True

detect_flag = False
detections = None
#classes = None
class Detect_Recognition_Processor:
	def __init__(self, classes):
		self.Q1 = queue.LifoQueue()
		self.Q2 = queue.LifoQueue()
		#self.CameraLoader = CameraLoader
		self.stopped = False
		#detect module
		self.detect_model = Darknet(opt.cfg, opt.detect_size)
		weights_path = opt.detect_weights_path
		if weights_path.endswith('.weigths'):
			load_weights(self.detect_model, weights_path)
		elif weights_path.endswith('.pt'):
			model_dict = torch.load(weights_path, map_location='cpu')
			self.detect_model.load_state_dict(model_dict['model'])
			del model_dict
		self.detect_model.to(device).eval()
		self.detect_sample_step = opt.detect_sample_step
		self.recognize_sample_step = opt.recognize_sample_step
		#recognition module
		self.recognize_model = I3D(opt.num_classes, input_size=opt.recognize_size)
		self.pretrained_model = torch.load(opt.recognize_weights_path, map_location=lambda storage, loc: storage)
		self.recognize_model.load_state_dict(self.pretrained_model)
		self.classes = classes
		self.recognition_mark = None
		
	def start(self):
		t1 = Thread(target=self.read_frame, args=())
		time.sleep(2)
		t1.start()
		t2 = Thread(target=self.detecting, args=())
		t2.start()
		#t3 = Thread(target=self.recognition, args=())
		#time.sleep(1)
		#t3.start()
		return self
	
	def read_frame(self):
		global detect_flag 
		global detections 
		#global classes
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
					continue
				orig_image = np.asanyarray(color_frame.get_data())
				detect_size = opt.detect_size
				img, _, _, _ = resize_square(orig_image, height=detect_size, color=(127.5,127.5,127.5)) #img is of shape (416,416,3)
				img = img[:, :, ::-1].transpose(2,0,1)  #bgr->rgb
				img = np.ascontiguousarray(img, dtype=np.float32)
				img /= 255.0
				self.Q1.put((img, orig_image))
				read_end = time.time()
				#print('read a frame cost time:%s seconds using cpu'%(read_end-read_start))
		except Exception as e:
			print(e)
		pass
	
	def detecting(self):
		detect_count = 0
		recognize_count = 0
		end_count = 0
		person_exists = False
		recognize_start_flag = False
		recognize_end_flag = False
		img_sequence = []
		Image_count = 0
		while True:
			detect_start = time.time()
			detections = None
			img, orig_image = self.get_camera_frame()
			img = torch.from_numpy(img).unsqueeze(0).to(device)
			if detect_count % self.detect_sample_step  == 0:
				predictions = self.detect_model(img)
				predictions = predictions[predictions[:, :, 4]>opt.conf_thres]
				if len(predictions)>0:
					detections = non_max_suppression(predictions.unsqueeze(0), opt.conf_thres, opt.nms_thres)
					if detections is not None:
						person_exists = True
				else:
					person_exists = False
			detect_end = time.time()
			print('detect a frame cost time:%s seconds using cpu'%(detect_end - detect_start))
			detect_count += 1
			detect_size = opt.detect_size
			#Person Detect
			if person_exists:
				image_with_box, box = draw_boxes(orig_image, detect_size, detections, self.classes)
				recognize_start_flag = True
				self.detect_sample_step = 1
				recognize_size = opt.recognize_size
				image_for_recognize = prep_image_for_recognize(orig_image, recognize_size, box)
			#No Person Detected
			else:
				if recognize_start_flag == False:
					pass
				else:
					end_count += 1
					if end_count==2:
						recognize_start_flag = False
						self.detect_sample_step = opt.detect_sample_step
						self.recognition_mark = None
						end_count = 0
					else:
						pass
			#Recognizing
			if recognize_start_flag:
				if recognize_count % self.recognize_sample_step == 0:
					img_sequence.append(torch.FloatTensor(image_for_recognize))
					recognize_count += 1
				if recognize_count % (self.recognize_sample_step * opt.temporal_sample_length) == 0:
					assert len(img_sequence)==opt.temporal_sample_length, 'got a wrong length of image sequence '
					video_tensor = torch.stack(img_sequence, 0)
					self.Q2.put((video_tensor, orig_image))		
					img_sequnece = []
			#Show Image
			Image_count += 1
			if opt.show:
				if person_exists:
					if self.recognition_mark is not None:
						cv2.putText(image_with_box, self.recognition_mark, (10,50), cv2.FONT_HERSHEY_TRIPLEX, 1,(0,0,255),1)
					else:
						pass
					cv2.imshow('recognition', image_with_box)  #show Image with both box and classes label
					cv2.imwrite('data/output/img%s.jpg'%Image_count, image_with_box)
				else:
					cv2.putText(orig_image, 'No person', (10,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 1)
					cv2.imshow('recognition', orig_image)#show Image with mark 'No Person'
					cv2.imwrite('data/output/img%s.jpg'%Image_count, orig_image)
				cv2.waitKey(1)
	
	def recognition(self):
		while True:
			recognition_start = time.time()
			video_tensor, img = self.get_video_tensor()
			predicted_label, confidence = model_test(video_tensor, self.recognize_model)
			label = predicted_label.item()
			self.recognition_mark = self.classes[label]
			recognition_end= time.time()
			print('recognition one time cost :%s'%(recognition_end - recognition_start))
	def get_camera_frame(self):
		return self.Q1.get()

	def get_video_tensor(self):
		return self.Q2.get()
'''	
class RecognizeProcessor:
	def __init__(self, DetectPocessor, classes):
		self.DetectProcessor = DetectProcessor
		self.recognize_model = I3D(opt.num_classes, input_size=opt.recognize_size)
		self.pretrained_model = torch.load(opt.recognize_weights_path, map_location=lambda storage, loc: storage)
		self.recognize_model.load_state_dict(self.pretrained_model)
		self.classes = classes
		
	def start(self):
		t3 = Thread(target=self.recognize, args=())
		t3.start()
		return self
	
	def recognize(self):
		global recognition_mark
		while True:
			video_tensor, img = self.DetectProcessor.getitem()
			predicted_label, confidence = model_test(video_tensor, self.recognize_model)
			label = predicted_label.item()
			recognition_mark = self.classes[label]
'''		
		
		
		
		
				
				
				
				
				