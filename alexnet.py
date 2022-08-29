#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 21:46
# @Author  : 陈伟峰
# @Site    : 
# @File    : alexnet.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def getModel():
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=96,kernel_size=11,strides=4,input_shape=(32,32,3),padding='same',activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='same'))
    model.add(keras.layers.Conv2D(filters=256,kernel_size=5,strides=1,padding='same',activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='same'))
    model.add(keras.layers.Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(keras.layers.Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096,activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096,activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10,activation='softmax'))   #修正的输出
    return model
def train_history(model_train,train,val):
    plt.plot(model_train.history[train])
    plt.plot(model_train.history[val])
    plt.title('Train History')
    plt.xlabel('epoch')
    plt.ylabel(train)
    plt.legend(['train','validation'],loc='upper left')

if __name__ == '__main__':

    (x_train_image,y_train_label),(x_test_image,y_test_label) = tf.keras.datasets.cifar10.load_data()
    x_train_normalize = x_train_image.astype('float32')/255
    x_test_normalize = x_test_image.astype('float32')/255
    y_train_OneHot = tf.keras.utils.to_categorical(y_train_label)
    y_test_OneHot = tf.keras.utils.to_categorical(y_test_label)
    model = getModel()
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model_train = model.fit(x=x_train_normalize,y=y_train_OneHot,validation_split=0.2,epochs=10,batch_size=300,verbose=1)
    train_history(model_train,'loss','val_loss')
    scores = model.evaluate(x_test_normalize,y_test_OneHot,verbose=2)
    print(scores)