from deep_neural_network.digital_gesture_recognition.load_model import *
import tensorflow as tf
import numpy as np


def load_model():
	with tf.gfile.GFile('model_only_pc//./digital_gesture.pb', "rb") as f:  #读取模型数据
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read()) #得到模型中的计算图和数据
	with tf.Graph().as_default() as graph:  # 这里的Graph()要有括号，不然会报TypeError
		tf.import_graph_def(graph_def, name="")  # 导入模型中的图到现在这个新的计算图中，不指定名字的话默认是 import
		# for op in graph.get_operations():  # 打印出图中的节点信息
		# 	print(op.name, op.values())
	return graph

def predict(graph):
	im = Image.open("dataset//new_pic//test6.jpg")
	mat = np.asarray(im.convert('RGB'))  # 原始图片
	mat = mat.reshape(1,64,64,3)
	mat = mat / 255.
	# keep_prob = graph.get_tensor_by_name("keep_prob:0")
	x = graph.get_tensor_by_name("input_x:0")
	outlayer = graph.get_tensor_by_name("outlayer:0")
	prob = graph.get_tensor_by_name("probability:0")
	predict = graph.get_tensor_by_name("predict:0")

	with tf.Session(graph=graph) as sess:
		# print(sess.run(output))
		np.set_printoptions(suppress=True)
		out, prob, pred = sess.run([outlayer, prob,predict],feed_dict={x:mat})
		print(out)
		print(prob)
		print(pred)


if __name__=="__main__":
	graph = load_model()
	# for op in graph.get_operations():  # 打印出图中的节点信息
	# 	print(op.name, op.values())
	predict(graph)