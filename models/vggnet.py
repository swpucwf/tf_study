#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/30 11:33
# @Author  : 陈伟峰
# @Site    : 
# @File    : vggnet.py
# @Software: PyCharm
from tensorflow import keras
import tensorflow as tf

def get_model():
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,input_shape=(32,32,3),padding='same',activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='same'))
    model.add(keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='same'))
    model.add(keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='same'))
    model.add(keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='same'))
    model.add(keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096,activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096,activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10,activation='softmax'))
    return model

if __name__ == '__main__':
    model = get_model()
    print(model.summary())