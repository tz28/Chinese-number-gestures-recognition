import os
from PIL import Image
import numpy as np
import h5py
import tensorflow as tf
import scipy.misc
import time

#压缩图片,把图片压缩成64*64的
def resize_img():
	dirs = os.listdir("split_pic//6")
	for filename in dirs:
		im = tf.gfile.FastGFile("split_pic//6//{}".format(filename), 'rb').read()
		# print("正在处理第%d张照片"%counter)
		with tf.Session() as sess:
			img_data = tf.image.decode_jpeg(im)
			image_float = tf.image.convert_image_dtype(img_data, tf.float32)
			resized = tf.image.resize_images(image_float, [64, 64], method=3)
			resized_im = resized.eval()
			# new_mat = np.asarray(resized_im).reshape(1, 64, 64, 3)
			scipy.misc.imsave("resized_img6//{}".format(filename),resized_im)

#图片转h5文件
def image_to_h5():
	dirs = os.listdir("resized_img")
	Y = [] #label
	X = [] #data
	print(len(dirs))
	for filename in dirs:
		label = int(filename.split('_')[0])
		Y.append(label)
		im = Image.open("resized_img//{}".format(filename)).convert('RGB')
		mat = np.asarray(im) #image 转矩阵
		X.append(mat)

	file = h5py.File("dataset//data.h5","w")
	file.create_dataset('X', data=np.array(X))
	file.create_dataset('Y', data=np.array(Y))
	file.close()

	#test
	# data = h5py.File("dataset//data.h5","r")
	# X_data = data['X']
	# print(X_data.shape)
	# Y_data = data['Y']
	# print(Y_data[123])
	# image = Image.fromarray(X_data[123]) #矩阵转图片并显示
	# image.show()


if __name__ == "__main__":
	# print("start.....: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
	# resize_img()
	# print("end....: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
	image_to_h5()