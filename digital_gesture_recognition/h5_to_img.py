import matplotlib.pyplot as plt
from machine_learning.deep_neural_network.digital_gesture_recognition.cnn import load_dataset


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# 训练集
index = 1
for i in range(X_train_orig.shape[0]):
    # plt.imshow(X_train_orig[i])
    plt.imsave(fname = "pic//train//{}_{}.png".format(Y_train_orig[0][i],str(index)), arr = X_train_orig[i])
    index += 1

#测试集
for i in range(X_test_orig.shape[0]):
    # plt.imshow(X_train_orig[i])
    plt.imsave(fname = "pic//test//{}_{}.png".format(Y_test_orig[0][i],str(index)), arr = X_test_orig[i])
    index += 1