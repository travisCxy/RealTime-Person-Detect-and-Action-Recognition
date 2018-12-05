import os
import time
import torch
import numpy as np
import sys
sys.path.append('..')
from opt import opt
from dataloader import VideoLoader, CameraRecognition ,RecognitionLoader, DataWriter
from class_list.get_classes_list import get_classes_list
from detect_recognize import CameraLoader, Detect_Recognition_Processor #RecognizeProcessor
from DR import Read_Detect_Recognition_Processor
from videoDR import Video_Detect_Recognition_Processor
from yolo.yolo_utils.utils import *

def Demo(opt):
	if opt.mode == 'video':
		if opt.modality == 'R':
			video_loader = VideoLoader(opt.video_path).start()
			(fourcc, fps, w, h) = video_loader.videoinfo()
			#load recognition loader
			print('Loading model...')
			sys.stdout.flush()
			Recognition= RecognitionLoader(video_loader, recognition_classes, opt.temporal_sample_length, step=opt.recognize_sample_step).start()
			#data writer
			if opt.save_video:
				if not os.path.exists(opt.out_path):
					os.makedirs(opt.out_path)
				save_path = os.path.join(opt.out_path, 'VideoDemo.avi')
				writer = DataWriter(Recognition, savepath=save_path, save_video=opt.save_video, vis=opt.show, fps=fps, frameSize=(w, h)).start()
				print('output video has been saved in %s'%save_path)
				writer.stop()
		elif opt.modality == 'DR':
			if opt.save_video:
				if not os.path.exists(opt.out_path):
					os.makedirs(opt.out_path)
				out_path = os.path.join(opt.out_path, 'result.avi')
				video_DR = Video_Detect_Recognition_Processor(opt.video_path, out_path, detect_classes, recognition_classes).start()		
				video_DR.read_detect()
	if opt.mode == 'camera':
		if opt.modality == 'R':
			CameraRecognitionLoader = CameraRecognition(classes).start()
		elif opt.modality== 'DR':
			#Camera = CameraLoader().start()
			#Detect = Detect_Recognition_Processor(classes).start()
			#Recognize = RecognizeProcessor(Detect, classes).start()
			DR = Read_Detect_Recognition_Processor(detect_classes, recognition_classes).start()
			DR.read_detect()

if __name__ == "__main__":
	if opt.dataset =='NTU':
		opt.num_classes = 60
		recognition_classes = get_classes_list('class_list/NTUClasses.txt', 'NTU') 
	elif opt.dataset == 'UCF':
		opt.num_classes = 101
		recognition_classes = get_classes_list('class_list/UCFClasses.txt', 'UCF')
		
	detect_classes = load_classes(opt.class_path)
	Demo(opt)
	
