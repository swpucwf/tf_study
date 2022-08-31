#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/30 11:06
# @Author  : 陈伟峰
# @Site    : 
# @File    : lenet_5.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow import keras

def lenet():
    model = tf.keras.models.Sequential()   #模型创建
    model.add(keras.layers.Conv2D(filters=6,kernel_size=3,strides=1,input_shape=(32,32,3)))
    model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Conv2D(filters=16,kernel_size=3,strides=1))
    model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(120,activation='relu'))
    model.add(keras.layers.Dense(84,activation='relu'))
    model.add(keras.layers.Dense(10,activation='softmax'))
    return model

def lenet_2():
    model = tf.keras.models.Sequential([
        keras.layers.Conv2D(filters=6,kernel_size=3,strides=1,input_shape=(32,32,3)),
        keras.layers.MaxPooling2D(pool_size=2,strides=2),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=16,kernel_size=3,strides=1),
        keras.layers.MaxPooling2D(pool_size=2,strides=2),
        keras.layers.ReLU(),
        keras.layers.Flatten(),
        keras.layers.Dense(120,activation='relu'),
        keras.layers.Dense(84,activation='relu'),
        keras.layers.Dense(10,activation='softmax')]
    )
    return model


# model= lenet()
model= lenet_2()
print(model.summary())