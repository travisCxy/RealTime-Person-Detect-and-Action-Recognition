import cv2
import torch
import sys
sys.path.append('..')
from net.I3D import I3D
from scipy.misc import imread


cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

image_path = 'example/video_R'
image_sequences = []
video_tensors = []
image_file_dir = os.listdir(image_path)
image_file_dir.sort()
for idx, file in enumerate(image_file_dir):
	image = imread(file)	#(H,W,3)
	cv2.imshow('a', image)
	cv2.waitKey(1)
	image_sequences.append(torch.from_numpy(image.copy())) 
	if len(image_sequences)==32:
		video_tensor = torch.stack(image_sequences, 0) #(32,h,w,3)
		video_tensor= np.array(video_tensor).transpose((0,3,1,2))#(32,3,h,w)
		video_tensor = torch.FloatTensor(video_tensor).unsqueeze(0).to(device)
		video_tensors.append(video_tensor)
model = I3D(num_classes=60, input_size=448)
model.load_state_dict(torch.load('weights/CS0.864'))
class_list = get_classes_list('class_list/NTUClasses.txt', 'NTU')
for video_tensor in video_tensors:
	label, confidence = model(video_tensor)
	mark = class_list[label]
	print('Class:%s  Confidence:%.3f'%(mark, confidence))


