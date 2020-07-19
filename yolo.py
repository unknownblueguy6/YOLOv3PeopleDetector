import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import tensornets as nets
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import time


mypath = 'input'

inputs = tf.placeholder(tf.float32, [None, 416, 416, 3]) 
model = nets.YOLOv3COCO(inputs, nets.Darknet19)

classes={'0':'person'}
list_of_classes=[0]

with tf.Session() as sess:
	sess.run(model.pretrained())
	
	files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	#change the path to your directory or to '0' for webcam!ssize.empty() in function 'resize'
	print(files)
	for file in files:
		
		try:
			frame = cv2.imread('input/%s' % file)
		except:
			continue
		
		dims = frame.shape
		dims_scalefactor = [416/dims[0], 416/dims[1]]

		img=cv2.resize(frame,(416,416), interpolation = cv2.INTER_AREA)
		imge=np.array(img).reshape(-1,416,416,3)
		preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
		boxes = model.get_boxes(preds, imge.shape[1:3])
		boxes1=np.array(boxes)
		
		for j in list_of_classes: #iterate over classes
			count = 0
			if str(j) in classes:
				lab=classes[str(j)]
			if len(boxes1) !=0:
				for i in range(len(boxes1[j])): 
					box=boxes1[j][i] 
					#setting confidence threshold as 40%
					if boxes1[j][i][4]>=.40: 
						count += 1 
						x1, y1 = np.float32(box[0]/dims_scalefactor[1]), np.float32(box[1]/dims_scalefactor[0])
						x2, y2 = np.float32(box[2]/dims_scalefactor[1]), np.float32(box[3]/dims_scalefactor[0])
						cv2.rectangle(frame,(x1,y1), (x2,y2), (0,255,0), 3)
			print(lab,": ",count)
	      
		cv2.imwrite('output/%s' % file, frame)