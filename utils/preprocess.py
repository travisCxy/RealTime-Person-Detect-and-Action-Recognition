import torch
import cv2 
import numpy as np
	
def prep_image(img, inp_dim):
	""" prepare image for the CNN input
	"""
	orig_image = img
	w, h  = orig_image.shape[1], orig_image.shape[0]
	if w>h:
		scale = float(inp_dim)/float(h)
		new_image = np.array(cv2.resize(orig_image, (int(w*scale+1),inp_dim))).astype(np.float32)
	else:
		scale = float(inp_dim)/float(w)
		new_image = np.array(cv2.resize(orig_image, (inp_dim, int(h*scale+1)))).astype(np.float32)
	new_w, new_h = new_image.shape[1], new_image.shape[0]
	crop_w = int((new_w - inp_dim)/2)
	crop_h = int((new_h - inp_dim)/2)
	img = new_image[crop_h:(crop_h+inp_dim), crop_w:(crop_w+inp_dim),:]
	return img, orig_image


	
def prep_image_for_recognize(orig_image, recognize_size, boxes):
	x1_list = []
	x2_list = []
	y1_list = []
	y2_list = []
	for i in range(len(boxes)):
		x1_list.append(boxes[i][0])
		y1_list.append(boxes[i][1])
		x2_list.append(boxes[i][2])
		y2_list.append(boxes[i][3])
	x1 = min(x1_list)
	x2 = max(x2_list)
	y1 = min(y1_list)
	y2 = max(y2_list)
	w = x2 - x1
	h = y2 - y1
	assert w>=0
	assert h>=0
	a = int(max(w, h) * 3/2)
	pad_w = int((a - w)/2)
	pad_h = int((a - h)/2)
	x_start = int(max(x1-pad_w, 0))
	y_start = int(max(y1-pad_h, 0))
	#print(x_start, y_start)
	crop_image_with_person = orig_image[x_start:(x_start+a), y_start:(y_start+a), :]
	#print(crop_image_with_person.shape)
	image_for_recognize, _ = prep_image(crop_image_with_person, recognize_size)
	return image_for_recognize
'''
orig_image = cv2.imread('000010.jpg')
boxes=[(50,100,230,240)]
img = prep_image_for_recognize(orig_image, 448, boxes)
print(img.shape)
cv2.imwrite('img.jpg', img)
'''
