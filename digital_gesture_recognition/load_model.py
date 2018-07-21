import tensorflow as tf
from machine_learning.deep_neural_network.digital_gesture_recognition import  cnn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import scipy.misc

# load trained parameters
def load_parameters():
	W_conv1 = tf.get_variable("W_conv1",shape = [5,5,3,32])
	b_conv1 = tf.get_variable("b_conv1", shape = [32])
	W_conv2= tf.get_variable("W_conv2", shape=[5, 5, 32, 64])
	b_conv2 = tf.get_variable("b_conv2", shape=[64])
	W_fc1 = tf.get_variable("W_fc1", shape = [16*16*64, 100])
	b_fc1 = tf.get_variable("b_fc1", shape = [100])
	W_fc2 = tf.get_variable("W_fc2", shape=[100, 11])
	b_fc2 = tf.get_variable("b_fc2", shape=[11])

	parameters = {}
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, "model//./cnn_model.ckpt")
		# print(W_conv1.eval())
		parameters["W_conv1"] = W_conv1.eval()
		parameters["b_conv1"] = b_conv1.eval()
		parameters["W_conv2"] = W_conv2.eval()
		parameters["b_conv2"] = b_conv2.eval()
		parameters["W_fc1"] = W_fc1.eval()
		parameters["b_fc1"] = b_fc1.eval()
		parameters["W_fc2"] = W_fc2.eval()
		parameters["b_fc2"] = b_fc2.eval()

	return parameters


def predict(parameters, X):
	W_conv1 = parameters["W_conv1"]
	b_conv1 = parameters["b_conv1"]
	W_conv2 = parameters["W_conv2"]
	b_conv2 = parameters["b_conv2"]
	W_fc1 = parameters["W_fc1"]
	b_fc1 = parameters["b_fc1"]
	W_fc2 = parameters["W_fc2"]
	b_fc2 = parameters["b_fc2"]

	x = tf.placeholder(tf.float32, [1, 64, 64, 3])
	z1 = tf.nn.relu(cnn.conv2d(x, W_conv1) + b_conv1)
	maxpool1 = cnn.max_pool_2x2(z1)
	z2 = tf.nn.relu(cnn.conv2d(maxpool1, W_conv2) + b_conv2)
	maxpool2 = cnn.max_pool_2x2(z2)
	maxpool2_flat = tf.reshape(maxpool2, [-1, 16 * 16 * 64])
	z_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, W_fc1) + b_fc1)
	logits = tf.matmul(z_fc1, W_fc2) + b_fc2
	logits = tf.nn.softmax(logits)
	c = tf.argmax(logits, 1)

	with tf.Session() as sess:
		prediction, logit = sess.run([c,logits], feed_dict={x: X})
	print("=======================")
	np.set_printoptions(suppress=True)
	print(logit)
	print(prediction)
	return prediction


#convert image to matrix
def img_to_mat(picname):
	im = Image.open("dataset//new_pic//{}".format(picname))
	mat = np.asarray(im.convert('RGB')) #原始图片
	# im.show()
	#新图片
	with tf.Session() as sess:
		image_float = tf.image.convert_image_dtype(im, tf.float32)
		resized = tf.image.resize_images(image_float, [64, 64], method=3)
		resized_im = resized.eval()
		new_mat = np.asarray(resized_im).reshape(1, 64, 64, 3)
		# print(new_mat)
		# scipy.misc.imsave("dataset//new_pic//test.png",resized_im)
	return mat, new_mat
	# new_image = im.resize((64,64))
	# new_image.show()
	# in_img = new_image.convert('RGB')
	# new_mat = np.asarray(in_img).reshape(1,64,64,3)  # image 转矩阵
	# # print(new_mat)
	# new_mat = new_mat / 255.
	# print(new_mat)

def display_result(mat, prediction):
	im = Image.fromarray(mat)#convert matrix to mat
	draw = ImageDraw.Draw(im)
	font = ImageFont.truetype('C:/windows/fonts/simhei.ttf', 150)
	draw.text((100, 100), "识别结果: {}".format(str(prediction)), fill= '#FF0000', font=font)
	im.show()

if __name__ == "__main__":
	mat, new_mat = img_to_mat("test.jpg")
	parameters = load_parameters()
	prediction = predict(parameters, new_mat)
	display_result(mat, prediction)
