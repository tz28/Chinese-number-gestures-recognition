import tensorflow as tf

def load_model():
	with tf.gfile.GFile('model_50_200_c3//./digital_gesture.pb', "rb") as f:  #读取模型数据
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read()) #得到模型中的计算图和数据
	with tf.Graph().as_default() as graph:  # 这里的Graph()要有括号，不然会报TypeError
		tf.import_graph_def(graph_def, name="")  # 导入模型中的图到现在这个新的计算图中，不指定名字的话默认是 import
		# for op in graph.get_operations():  # 打印出图中的节点信息
		# 	print(op.name, op.values())
	return graph
import numpy as np
from PIL import Image




if __name__=="__main__":
	graph = load_model()
	for op in graph.get_operations():  # 打印出图中的节点信息
		print(op.name, op.values())

	# im = Image.open("dataset//new_pic//hc11.jpg")
	# # im.show()
	# mat = np.asarray(im.convert('RGB'))  # 原始图片
	# mat = mat / 255.
	# mat = np.reshape(mat,(64*64*3,1))
	# print(mat[0:30,:])