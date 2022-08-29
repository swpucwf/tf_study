#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/26 19:42
# @Author  : 陈伟峰
# @Site    : 
# @File    : 01-单层神经网络.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def convert_onehot_label(train_labels):
    max_label,min_label = tf.reduce_max(train_labels),tf.reduce_min(train_labels)
    print(max_label,min_label)
    T = np.zeros((train_labels.size, 10))   #创建数组长度为10的全0标签
    for idx,row in enumerate(T):
        row[train_labels[idx]] =1
    return T

def sigmoid(x):   #激活函数
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):   #激活函数导数，用于在反向传播过程中求偏导
    f = sigmoid(x)
    return f * (1 - f)

train_data,test_data = tf.keras.datasets.mnist.load_data()
print(train_data[0].shape)
print(train_data[1].shape)
print(test_data[0].shape)
print(test_data[1].shape)

train_images,train_labels = train_data[0],train_data[1]
test_images,test_labels = test_data[0],test_data[1]

train_image = train_images.astype(np.float32)/255.0   #图像归一化处理
test_image = test_images.astype(np.float32)/255.0

#标签one-hot编码转换
train_labels = convert_onehot_label(train_labels)
test_labels = convert_onehot_label(test_labels)

w1 = np.random.normal(0.0, pow(784, -0.5), (784, 10))   #输入784，输出10
b1 = np.zeros((10,1))+0.0001   #偏置,初始化为0.0001
lr = 0.001   #学习率
epoch = 100   #训练次数
total_loss = []   #存放损失值，便于画图
stop_epoch = 0   #记录训练停止时的训练次数

for i in range(epoch):   #模型训练
    accurate_number = 0   #记录在一次训练里有多少张图片被预测正确
    for single_train_image,single_train_label in zip(train_image,train_labels):   #读取单张图像数据及其对应的标签
        epoch_loss = 0
        single_train_image = single_train_image.reshape(784)
        # print(single_train_image.shape)
        single_train_image = np.expand_dims(single_train_image, axis=1)   #将(784,)增添维度至(784,1)以满足矩阵运算需求
        single_train_label = np.expand_dims(single_train_label, axis=1)


        out1 = np.dot(w1.T,single_train_image)+b1   #前向传播 y=wx+b

        out1_ = sigmoid(out1)   #激活函数输出

        loss = (single_train_label-out1_)**2   #均方差损失函数
        epoch_loss += loss.mean()   #记录loss值

        grad_y = single_train_label-out1_   #对应∂y/∂loss项，常数项可被省略
        grad_w1 = 2*np.dot(single_train_image,(grad_y*deriv_sigmoid(out1)).T)   #对应∂loss/∂w的计算公式
        grad_b1 = (grad_y * deriv_sigmoid(out1))   #对应∂loss/∂b的计算公式


        w1 += lr * grad_w1   #w参数更新
        b1 += lr * grad_b1   #b参数更新

        if np.argmax(out1_) == np.argmax(single_train_label):   #判断预测和标签是否一致
            accurate_number+=1
    train_accuracy = accurate_number/len(train_image)   #计算训练一次的准确率
    print("train accuracy is:%f"%train_accuracy)
    total_loss.append(epoch_loss/len(train_image))   #记录每次训练的平均损失率
    stop_epoch+=1
    if train_accuracy > 0.90:   #当准确率大于0.90时，训练终止
        break

plt.plot([i for i in range(stop_epoch)],total_loss)   #横轴为训练次数，纵轴为损失值
plt.show()