#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 20:48
# @Author  : 陈伟峰
# @Site    : 
# @File    : 图像分类.py
# @Software: PyCharm
import tensorflow as tf

x = tf.random.normal([1,4,4,1])   # 构建批大小、图像尺寸、输入通道数分别为1，4x4和1的图像
w = tf.random.normal([4,4,1,3])   # 构建卷积核大小、输入及输出通道数分别为4x4、1和3的卷积核
out = tf.nn.conv2d(x,w,strides=(1,1,1,1),padding='SAME')
print(out)

x = tf.random.normal([1,4,4,1])
out = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=2,padding='VALID')   #步长为2，不进行补0
print(out)

x = tf.random.normal([1,4,4,1])   #构造图像结构
layer = tf.keras.layers.Conv2D(filters=3,kernel_size=4,strides=1,padding='SAME')   #调用卷积层类
out = layer(x)
print(out.shape)
layer = tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='SAME')   #调用最大池化层类
out = layer(x)
print(out.shape)
layer = tf.keras.layers.AveragePooling2D(pool_size=2,strides=2,padding='SAME')   #调用平均池化层类
print(out.shape)