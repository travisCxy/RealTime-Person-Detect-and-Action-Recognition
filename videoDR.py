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

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
detect_flag = False
detections = None

class Video_Detect_Recognition_Processor:
	def __init__(self, video_path, out_path, detect_classes, recognition_classes):
		
		#define a video stream and Q
		self.stream = cv2.VideoCapture(video_path)
		self.stopped = False
		assert self.stream.isOpened(), 'Cannot capture source'
		self.Q = queue.LifoQueue()
		fps = self.stream.get(cv2.CAP_PROP_FPS)
		w = int(self.stream.get(3))
		h = int(self.stream.get(4))
		if opt.save_video:
			self.out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), int(fps), (w,h), 1)
		#load detect module
		load_yolo_start = time.time()
		self.detect_model = Darknet(opt.cfg, opt.detect_size)
		weights_path = opt.detect_weights_path
		if weights_path.endswith('.weigths'):
			load_weights(self.detect_model, weights_path)
		elif weights_path.endswith('.pt'):
			model_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)
			self.detect_model.load_state_dict(model_dict['model'])
			del model_dict
		self.detect_model.to(device).eval()
		self.detect_sample_step = opt.detect_sample_step
		self.recognize_sample_step = opt.recognize_sample_step
		load_yolo_end = time.time()
		print('load yolo cost :%.4f using gpu'%(load_yolo_end - load_yolo_start))
		
		#load recognition module
		self.recognize_model = I3D(opt.num_classes, input_size=opt.recognize_size)
		self.pretrained_model = torch.load(opt.recognize_weights_path, map_location=lambda storage, loc: storage)
		self.recognize_model.load_state_dict(self.pretrained_model)
		self.detect_classes = detect_classes
		self.recognition_classes = recognition_classes
		self.recognition_mark = None
		load_i3d_end = time.time()
		print('load i3d cost :%.4f using gpu'%(load_i3d_end - load_yolo_end))
		
	def start(self):
		t = Thread(target=self.recognition, args=())
		time.sleep(2)
		t.start()
		return self
	
	def read_detect(self):
		global detect_flag 
		global detections 
		detect_count = 0
		recognize_count = 0
		end_count = 0
		person_exists = False
		recognize_start_flag = False
		recognize_end_flag = False
		img_sequence = []
		count = 0

		while True:
				if self.stop():
					return
				read_start = time.time()
				grabbed, orig_image = self.stream.read()
				if not grabbed:
					print('All frames have been processed')
					self.stop()
					if opt.save_video:
						self.out_writer.release()
					break
				detect_size = opt.detect_size
				img, _, _, _ = resize_square(orig_image, height=detect_size, color=(127.5,127.5,127.5)) #img is of shape (416,416,3)
				img = img[:, :, ::-1].transpose(2,0,1)  #bgr->rgb
				img = np.ascontiguousarray(img, dtype=np.float32)
				img /= 255.0
				read_end = time.time()
				#print('read a frame cost:%.3f'%(read_end - read_start))
				detect_start = time.time()
				detections = None
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
				#print('detect a frame cost time:%.3f seconds using cpu'%(detect_end - detect_start))
				detect_count += 1
				detect_size = opt.detect_size
				orig_image_copy = orig_image.copy()   #deep copy
				#Person Detect
				if person_exists:
					image_with_box, box = draw_boxes(orig_image, detect_size, detections, self.detect_classes)
					recognize_start_flag = True
					self.detect_sample_step = 1
					recognize_size = opt.recognize_size
					image_for_recognize = prep_image_for_recognize(orig_image_copy, recognize_size, box)
					if (recognize_count % self.recognize_sample_step) == 0:
						img_sequence.append(torch.FloatTensor(image_for_recognize))
						if not os.path.exists('data/test/%s'%count):
							os.makedirs('data/test/%s'%count) 
					recognize_count += 1
					if recognize_count % (self.recognize_sample_step * opt.temporal_sample_length) == 0:
						print('recognizing')
						assert len(img_sequence)==opt.temporal_sample_length, 'got a wrong length of image sequence '
						video_tensor = torch.stack(img_sequence, 0)
						self.Q.put((video_tensor, orig_image))		
						img_sequence = []
						count += 1
					if self.recognition_mark is not None:
						cv2.putText(image_with_box, self.recognition_mark, (10,50), cv2.FONT_HERSHEY_TRIPLEX, 1,(0,0,255),1)
					if opt.save_video:
						self.out_writer.write(image_with_box)
				#No Person Detected
				else:
					if recognize_start_flag == False:
						cv2.putText(orig_image, 'No person', (10,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 1)
						if opt.save_video:
							self.out_writer.write(orig_image)
					else:
						end_count += 1
						if end_count==2:
							recognize_start_flag = False
							self.detect_sample_step = opt.detect_sample_step
							self.recognition_mark = None
							end_count = 0

	def recognition(self):
		while True:
			video_tensor, img = self.get_video_tensor()
			recognition_start = time.time()
			predicted_label, confidence = model_test(video_tensor, self.recognize_model)
			label = predicted_label.item()
			self.recognition_mark = self.recognition_classes[label]
			print(self.recognition_mark)
			recognition_end= time.time()
			with open('result.txt', 'a') as file:
				file.write('Video Detect and Recognition label:'+str(self.recognition_mark)+'\n')
			#print('recognition one time cost :%s'%(recognition_end - recognition_start))

	def stop(self):
		self.stopped = True
		
	def get_video_tensor(self):
		return self.Q.get()
	
