import torch
import numpy as np
import pandas as pd

def read_box_info(box_info_path):
	"""
	box_info is a dict, where key is the number of frame.
	for each key, the frame_box_info is also a dict including all the coordinates of box
    each frame_box_info contain four keys 'x1', 'y1', 'x2', 'y2', each value of those keys is a list
	"""
	
	box_info_csv = pd.read_csv(box_info_path, delimiter=' ', header=None)
	box_info = {}
	x1 = {}
	y1 = {}
	x2 = {}
	y2 = {}
	for idx in range(len(box_info_csv)):
		key = box_info_csv.iloc[idx, 0].split('rgb/')[1].split('.')[0]
		x1[key] = []
		x2[key] = []
		y1[key] = []
		y2[key] = []
		box_info[key] = {}
	
	for idx in range(len(box_info_csv)):
		key = box_info_csv.iloc[idx, 0].split('rgb/')[1].split('.')[0]
		frame_x1, frame_y1, frame_x2, frame_y2 = box_info_csv.iloc[idx, 1],  box_info_csv.iloc[idx, 2], box_info_csv.iloc[idx, 3], box_info_csv.iloc[idx, 4]
		frame_x1, frame_y1, frame_x2, frame_y2 = max(frame_x1, 0), max(frame_y1, 0), max(frame_x2, 0), max(frame_y2, 0)
		x1[key].append(frame_x1)
		x2[key].append(frame_x2)
		y1[key].append(frame_y1)
		y2[key].append(frame_y2)
	
	for idx in range(len(box_info_csv)):
		key = box_info_csv.iloc[idx, 0].split('rgb/')[1].split('.')[0]
		box_info[key]['x1'] = x1[key]
		box_info[key]['y1'] = y1[key]
		box_info[key]['x2'] = x2[key]
		box_info[key]['y2'] = y2[key]
	
	return box_info
	
"""example:
box_info_path = 'Activity_label/S001C001P001R001A001_rgb.txt'
box_info = read_box_info(box_info_path)
y1 = box_info['0001']['x1']
print('x1 of frame 0001.jpg is:', y1)
"""
		

class crop_by_box(object):
	def __call___(self, sample):
		video, label, rand_frames, box_info = sample['video'], sample['label'], sample['rand_frames'], sample['box_info']
		new_video = np.zeros((video.shape[0], opt.input_size, opt.input_size, 3))	# (32, h, w, 3) rgb
		for idx, frame in enumerate(rand_frames):
			orig_image = video[idx, :, :, :]
			key = '%4d'%frame
			frame_box_info = box_info[key]
			x1_list, y1_list, x2_list, y2_list = frame_box_info['x1'], frame_box_info['y1'], frame_box_info['x2'], frame_box_info['y2']
			x1 = min(x1_list)
			x2 = max(x2_list)
			y1 = min(y1_list)
			y2 = max(y2_list)
			w = x2 - x1
			h = y2 - y1
			assert w>=0
			assert h>=0
			a = int(max(w, h) * opt.scale)
			pad_w = int((a - w)/2)
			pad_h = int((a - h)/2)
			x_start = int(max(x1-pad_w, 0))
			y_start = int(max(y1-pad_h, 0))
			crop_image_with_person = orig_image[x_start:(x_start+a), y_start:(y_start+a), :]
			image_for_recognize, _ = prep_image(crop_image_with_person, opt.input_size)
			image_for_recognize = image_for_recognize[:, :, ::-1]
			new_video[idx,:,:,:] = image_for_recognize
		return {'video':new_video, 'label':label}
	
class ToTensor(object):
	def __call__(self, sample):
		video, label = sample['video'], sample['label']
		video = video.transpose((0, 3, 1, 2)) #(32, 3, h, w)
		label = np.array([label])
		return {'video': torch.from_numpy(video), 'label':torch.FloatTensor(label)}
			
			
			
			