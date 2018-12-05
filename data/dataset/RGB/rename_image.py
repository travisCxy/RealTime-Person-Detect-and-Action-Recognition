import os

dir='img_sequence'
image_file_list = os.listdir(dir)
image_file_list.sort()
length = len(image_file_list)
count = 0
for i, file in  enumerate(image_file_list):
	print(file)
	os.rename(os.path.join(dir,file), os.path.join(dir, 'img%d.jpg'%count))
	count+=1
