import os
import torch
import cv2
import sys
import time
sys.path.append('..')

import time
import math
import numpy as np
import pyrealsense2 as rs
import threading
import queue
from recognition.net.I3D import I3D
from PIL import Image, ImageDraw
from threading import Thread
from utils.utils import *
from utils.preprocess import *
from threading import Thread
from opt import opt

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

##############################################Loader for clipped-Video recognition#############################
class VideoLoader:
	def __init__(self, video_path, queueSize=4096):
		# initialize the file video stream
		self.stream = cv2.VideoCapture(video_path)
		assert self.stream.isOpened(), 'Cannot capture source'
		
		# indicate if the thread should be stopped
		self.stopped = False
		
		# initialize the queue use to store frames read from stream
		self.Q = queue.Queue(maxsize=queueSize)
		self.datalen = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))

	def start(self):
		'''
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		'''
		self.update()
		return self
	
	def update(self):
		# keep looping infinitely 
		while True:
			# ensure the queue is not full
			if not self.Q.full():
				# read frame from video stream
				(grabbed, frame) = self.stream.read()
				# if grabbed is False, means we have reached the end of the video file
				if not grabbed:
					self.stop()
					return
				#process the frame
				inp_dim = int(opt.recognize_size)
				img, orig_img = prep_image(frame, inp_dim)
				img = img[:, :, ::-1]
				self.Q.put((img, frame))
			else:
				with self.Q.mutex:
					self.Q.queue.clear()
			
	def videoinfo(self):
		fourcc = int(self.stream.get(cv2.CAP_PROP_FOURCC))
		fps = self.stream.get(cv2.CAP_PROP_FPS)
		w = int(self.stream.get(3))
		h = int(self.stream.get(4))
		return (fourcc, fps, w, h)
	
	#def stream(self):
		#return self.stream
	
	def getitem(self):
		#return next frame in the queue
		return self.Q.get()
			
	def stop(self):
		# stop the thread
		self.stopped = True

	def len(self):
		return self.datalen
	
	
class RecognitionLoader:
	def __init__(self, CameraLoader, classes, temporalSize=32, queueSize=1024, step=opt.recognize_sample_step):
		self.CameraLoader = CameraLoader
		self.temporalSize = temporalSize
		self.datalen = self.CameraLoader.len()
		self.model = I3D(opt.num_classes, input_size=opt.recognize_size)
		pretrained_model = torch.load(opt.recognize_weights_path, map_location='cpu')
		self.model.load_state_dict(pretrained_model)
		
		self.step = step
		self.classes = classes

		self.Q = queue.Queue(maxsize=queueSize)	#存放处理过后的帧
		self.stream = self.CameraLoader.stream
		
	def start(self):
		'''
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		'''
		self.update()
		return self
		
	def update(self):
		# keep looping the whole dataset
		count = 0
		flag = False
		img_sequence = []
		for i in range(self.datalen):
			with torch.no_grad():
				img, frame= self.CameraLoader.getitem() #img is of shape (h,w,3)--(224,224,3)
				count += 1
				if count % self.step == 0:
					img_sequence.append(torch.from_numpy(img.copy()))
				if flag:
					cv2.putText(frame, self.classes[label], (10,50), cv2.FONT_HERSHEY_TRIPLEX, 1,(0,0,255),1)
				self.Q.put(frame)
				if count % (self.temporalSize*self.step) == 0:
					video_tensor = torch.stack(img_sequence, 0)
					predicted_label, confidence = model_test(video_tensor, self.model)
					label = int(predicted_label.item())
					img_sequence = []
					flag = True
	
	def getitem(self):
		return self.Q.get()
	
	def len(self):
		return self.Q.qsize()


class DataWriter:
	def __init__(self, RecognitionLoader, savepath='example/result.avi', save_video=False, vis=False,
			 fourcc=cv2.VideoWriter_fourcc(*'XVID') ,fps=25, frameSize=(640, 480)):
		self.stream = RecognitionLoader.stream
		if save_video:
			self.outstream = cv2.VideoWriter(savepath, fourcc, int(fps), (int(self.stream.get(3)), int(self.stream.get(4))), 1)
		self.save_video =save_video
		self.stopped = False
		self.RecognitionLoader = RecognitionLoader
		self.datalen = self.RecognitionLoader.len()
		self.vis = vis
	
	def start(self):
		'''
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		'''
		self.update()
		return self
	
	def update(self):
		for i in range(self.datalen):
			if self.stopped:
				if self.save_video:
					self.outstream.release()
				return
				
			frame = self.RecognitionLoader.getitem()
			if self.save_video:
				self.outstream.write(frame)
			if self.vis:
				cv2.imshow('ACTION RECOGNITION', frame)
				cv2.waitKey(30)
	def stop(self):
		self.stopped = True
		time.sleep(0.2)
	
	
			
				
			
##############################################Loader for only Real-Time recognition#############################		
flag=False
mark=None
class CameraRecognition:
	def __init__(self, classes):
		self.Q = queue.LifoQueue()
		self.stopped = False
		self.flag=False
		self.mark=None
		self.model = I3D(opt.num_classes, input_size=opt.recognize_size)
		pretrained_model = torch.load(opt.recognize_weights_path, map_location=lambda storage, loc: storage)
		self.model.load_state_dict(pretrained_model)
		self.classes = classes

	def start(self):
		t1 = Thread(target=self.read_camera, args=())
		#t1.daemon = True
		t1.start()
		#time.sleep(1)
		t2 = Thread(target=self.process, args=())
		#t2.daemon=True
		t2.start()
		return self
	
	def read_camera(self):
		print('thread to read camera:%s'%os.getpid())
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
		pipeline.start(config)
		count = 0
		img_sequence = []		
		try:
			while True:
				frames = pipeline.wait_for_frames()
				frames.keep()
				color_frame = frames.get_color_frame()
				if not color_frame:
					continue
				count += 1
				#print(count)
				color_frame = np.asanyarray(color_frame.get_data())
				inp_dim = opt.recognize_size
				img, orig_img = prep_image(color_frame, inp_dim)
				img = img[:,:,::-1] 	#bgr->rgb
				if count % opt.step == 0:
					img_sequence.append(torch.from_numpy(img.copy()))
				if count % (opt.temporal_sample_length * opt.step) == 0:
					assert len(img_sequence)==opt.temporal_sample_length
					video_tensor = torch.stack(img_sequence, 0)
					img_sequence = []
					self.Q.put((video_tensor, orig_img))
				if self.flag:
					cv2.putText(orig_img, self.mark, (10,50), cv2.FONT_HERSHEY_TRIPLEX, 1,(0,0,255),1)
				if opt.show:
					cv2.imshow('Action Recognition', orig_img)
					cv2.waitKey(1)
		except Exception as e:
			print(e)
		pass			
	
	def process(self):	
		print('thread to process image sequence:%s'%os.getpid())
		while True:
			video_tensor, img = self.getitem()
			start = time.time()
			predicted_label, confidence = model_test(video_tensor, self.model)
			end = time.time()
			print('model test:%s'%(end-start))
			label = predicted_label.item()
			self.flag = True
			self.mark = self.classes[label]

	def getitem(self):
		return self.Q.get()

	def stop(self):
		self.stop = True		
			
			
			
			
			
			
			
			
			
