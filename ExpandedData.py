# from PIL import Image
# import os
# #img = data.camera()
#
# data_dir = 'D:/python-project/CatVsDogRecong/train_image/0/'
# for fileName in os.listdir(data_dir):
# 	img = Image.open(os.path.join(data_dir, fileName))
# 	out90 = img.rotate(45) #旋转
# 	out180 = img.rotate(90) #旋转
# 	out270 = img.rotate(135) #旋转
# 	out90.save(data_dir+'rotate45_'+fileName)
# 	out180.save(data_dir+'rotate90'+fileName)
# 	out270.save(data_dir+'rotate135_'+fileName)
# 	#cv2.imwrite(data_dir+'rotate180'+fileName,img2)
# 	#cv2.imwrite(data_dir+'rotate270'+fileName,img3)
# 	#io.imsave(data_dir+'rotate90_'+fileName,img1)
# 	#io.imsave(data_dir+'rotate180_'+fileName,img2)
# 	#io.imsave(data_dir+'rotate270_'+fileName,img3)
#
# 	#img2=transform.rotate(img, 30,resize=True)  #旋转30度，同时改变大小
# 	#print(img2.shape)
import cv2
from math import *
import numpy as np
import os
data_dir='D:/python-project/CatVsDogRecong/train_image__/0_c/'

for fileName in os.listdir(data_dir):
	print(data_dir+fileName)
	img = cv2.imread(data_dir+fileName)

	height, width,channle = img.shape

	degree = 180
	# 旋转后的尺寸
	heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  # 这个公式参考之前内容
	widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

	matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

	matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
	matRotation[1, 2] += (heightNew - height) / 2

	imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
	cv2.imwrite(data_dir+str(degree)+fileName,imgRotation)
	# cv2.imshow("img", img)
	# cv2.imshow("imgRotation", imgRotation)
	# cv2.waitKey(0)


