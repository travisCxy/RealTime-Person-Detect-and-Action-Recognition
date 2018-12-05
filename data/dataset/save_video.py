import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse

parser =argparse.ArgumentParser('Save Video')
parser.add_argument('--time', default=4, type=int, help='video length')
opt = parser.parse_args()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
frame_count = int(opt.time * 30)

path = 'RGB/img_sequence'
if not os.path.exists(path):
	os.makedirs(path)
pipeline.start(config)
count = 0
try:
	while True:
		frames = pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		if not depth_frame or not color_frame:
			continue
		#convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		cv2.imwrite('RGB/img_sequence/img%s.jpg'%count,color_image)
		cv2.imshow('a', color_image)
		cv2.waitKey(30)
		count += 1
		if count == frame_count :
			break
		#if cv2.waitKey(1) == ord("q"):
			#break
		
except Exception as e:
	print(e)
pass
