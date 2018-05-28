#--------------------------开始在新图片中检测-------------------#
import tensorflow as tf
import cv2
import numpy as np

import matplotlib.pyplot as plt

#----------------定义网络中频繁使用的函数，将其重构-----------------#
#权重变量
def weight_variables(shape):
    weights = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(weights)

#偏置变量
def biase_variables(shape):
    biases = tf.constant(value=1.0, shape=shape)
    return tf.Variable(biases)

#卷积
def conv2d(x, W):
    '''计算卷积，x为输入层（shape=[-1,width,height,channel]）,
    W为f*f的共享权重矩阵shape=[f,f,in_layers_num, out_layers_num]，
    水平和垂直方向上的步长都为1'''
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="VALID")

#最大值池化
def max_pooling(x):
    '''计算最大值混合，x为输入层(一般是卷积结果)shape=[-1,width,height,channels]
    ksize为混合pooling的核大小2*2，水平和垂直方向上的步长都为2'''
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

def deepnn(x, keep_prop):
    '''定义深层卷积网络，包含了两个卷积-混合层和三个卷积层'''
    #step1:将原始一维得得数据转换成2维, 第一个表示样本数，第二三个是行列，最后一个是通道数
#     x = tf.reshape(x, shape=[-1, 40, 100, 1])
    #step2:定义第一的卷积-混合层
    with tf.name_scope("conv-pooling1"):
        W_conv1 = weight_variables([5,5,1,6])
        b_conv1 = biase_variables([6])
        ret_conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)  #计算卷积，并使用修正单元对卷积结果进一步处理
        ret_pooling1 = max_pooling(ret_conv1)  #执行混合操作

    #step3:定义第二个卷积-混合层
    with tf.name_scope("conv-pooling2"):
        W_conv2 = weight_variables([5,5,6,16])
        b_conv2 = biase_variables([16])
        ret_conv2 = tf.nn.relu(conv2d(ret_pooling1, W_conv2) + b_conv2)
        ret_pooling2 = max_pooling(ret_conv2)

    #step4:定义第三个卷积层
    with tf.name_scope("conv-pooling3"):
        W_conv3 = weight_variables([5,5,16,32])
        b_conv3 = biase_variables([32])
        ret_conv3 = tf.nn.relu(conv2d(ret_pooling2, W_conv3) + b_conv3)

    #step5:定义第四个卷积层
    with tf.name_scope("conv4"):
        W_conv4 = weight_variables([3,18,32,64])
        b_conv4 = biase_variables([64])
        ret_conv4 = tf.nn.relu(conv2d(ret_conv3, W_conv4) + b_conv4)

    #step6:定义第五个卷积层
    with tf.name_scope("conv5"):
        W_conv5 = weight_variables([1,1,64,1])
        b_conv5 = biase_variables([1])
        ret_conv5 = conv2d(ret_conv4, W_conv5) + b_conv5

    return ret_conv5
#导入图片
pic = cv2.imread("./datas/CarData/TestImages/test-100.pgm", 0)

size = pic.shape

img  = np.reshape(pic, (-1,size[0], size[1], 1))
#利用上面训练好的网络，开始在新的图片中检测

#申明输入数据和标签的占位符
x = tf.placeholder(dtype=tf.float32, shape=[None,None, None, 1], name="x-input")
labels = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y-output")

#申明弃权的占位符
keep_prop = tf.placeholder(dtype=tf.float32, name="kprob")

#创建分类模型
ret = deepnn(x, keep_prop)
#此时的返回值是 -1*1*1*1的， 为了得到方便运算的结果，这里将reshape
y = tf.reshape(ret, shape=[-1,1])

sess=tf.Session()
saver=tf.train.Saver()
saver.restore(sess,"./models/carDetect_model.ckpt")
result = sess.run(ret, feed_dict={x:img})

#将检测结果显示
pt1 = np.array([result.argmax()//result.shape[2], result.argmax()%result.shape[2]]) * 4
pt2 = pt1 + np.array([40, 100])

pic_2 = cv2.rectangle(pic, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 0, 2)

plt.imshow(pic_2, "gray")
plt.show()