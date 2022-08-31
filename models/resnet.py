#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/30 11:11
# @Author  : 陈伟峰
# @Site    : 
# @File    : resnet.py
# @Software: PyCharm
from tensorflow import keras


def identity_block(x,filters):
    x_input = x
    x = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), padding = 'same')(x)
    x = keras.layers.BatchNormalization(axis = 3)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), padding = 'same')(x)
    x = keras.layers.BatchNormalization(axis = 3)(x)
    x = keras.layers.Add()([x, x_input])   #与输入进行连接
    x = keras.layers.ReLU()(x)
    return x
def convolutional_block(x,filters):
    x_input = x
    x = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), padding = 'same')(x)
    x = keras.layers.BatchNormalization(axis = 3)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), padding = 'same')(x)
    x = keras.layers.BatchNormalization(axis = 3)(x)
    x_input = keras.layers.Conv2D(filters, kernel_size=(1,1), strides = (1,1))(x_input)
    x_input = keras.layers.BatchNormalization(axis = 3)(x_input)
    x = keras.layers.Add()([x, x_input])
    x = keras.layers.ReLU()(x)
    return x

def resNet():
    inputs = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Conv2D(64, (7, 7), strides = (2,2),padding='same')(inputs)
    x = keras.layers.BatchNormalization(axis = 3)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((3, 3), strides = (2,2))(x)
    x = convolutional_block(x,64)
    x = identity_block(x,64)
    x = identity_block(x,64)
    x = convolutional_block(x,128)
    x = identity_block(x,128)
    x = identity_block(x,128)
    x = identity_block(x,128)
    x = convolutional_block(x,256)
    x = identity_block(x,256)
    x = identity_block(x,256)
    x = identity_block(x,256)
    x = identity_block(x,256)
    x = identity_block(x,256)
    x = convolutional_block(x,512)
    x = identity_block(x,512)
    x = identity_block(x,512)
    x = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10,activation='softmax')(x)
    model = keras.models.Model(inputs,x)
    return  model

if __name__ == '__main__':
    model = resNet()
    print(model.summary())


