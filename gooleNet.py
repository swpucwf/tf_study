#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 22:13
# @Author  : 陈伟峰
# @Site    : 
# @File    : gooleNet.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# def Conv2d_BN(x,filter,kernel_size,strides=(1,1)):
#
#     x = keras.layers.Conv2D(filter,kernel_size,paddind="same",strides=strides,activation="relu")(x)
#     x = keras.layers.BatchNormlization(axis=3)(x)
#     return x
def Conv2d_BN(x,filter,kernel_size,strides=(1,1)):
    x = keras.layers.Conv2D(filter,kernel_size,padding='same',strides=strides,activation='relu')(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    return x

def Inception(x,filter):
    branch1x1 = Conv2d_BN(x,filter,(1,1), strides=(1,1))
    branch3x3 = Conv2d_BN(branch1x1,filter,(3,3), strides=(1,1))

    branch1x1 = Conv2d_BN(x,filter,(1,1),strides=(1,1))
    branch5x5 = Conv2d_BN(branch1x1,filter,(1,1),strides=(1,1))

    branch3x3_pooling = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branch1x1_pooling = Conv2d_BN(branch3x3_pooling,filter,(1,1),strides=(1,1))

    x = keras.layers.concatenate([branch1x1,branch3x3,branch5x5,branch1x1_pooling],axis=3)
    return x

def train_history(model_train,train,val):
    plt.plot(model_train.history[train])
    plt.plot(model_train.history[val])
    plt.title('Train History')
    plt.xlabel('epoch')
    plt.ylabel(train)
    plt.legend(['train','validation'],loc='upper left')

if __name__ == '__main__':
    '''
    数据集传入
    '''
    (x_train_image,y_train_label),(x_test_image,y_test_label) = tf.keras.datasets.cifar10.load_data()
    x_train_normalize = x_train_image.astype('float32')/255
    x_test_normalize = x_test_image.astype('float32')/255
    y_train_OneHot = tf.keras.utils.to_categorical(y_train_label)
    y_test_OneHot = tf.keras.utils.to_categorical(y_test_label)


    inputs = keras.layers.Input(shape=(32,32,3))
    x = Conv2d_BN(inputs,64,(7,7),strides=(2,2))
    x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Conv2d_BN(x,192,(3,3),strides=(1,1))
    x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,64)
    x = Inception(x,120)
    x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,128)
    x = Inception(x,128)
    x = Inception(x,128)
    x = Inception(x,132)
    x = Inception(x,208)
    x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,208)
    x = Inception(x,256)
    x = keras.layers.AveragePooling2D(pool_size=(7,7),strides=(1,1),padding='same')(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(1000,activation='relu')(x)
    x = keras.layers.Dense(10,activation='softmax')(x)
    x = tf.squeeze(x,axis=1)
    x = tf.squeeze(x,axis=1)


    model = keras.models.Model(inputs,x,name='inception')
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model_train = model.fit(x=x_train_normalize,y=y_train_OneHot,validation_split=0.2,epochs=50,batch_size=300,verbose=1)
    train_history(model_train,'loss','val_loss')
    scores = model.evaluate(x_test_normalize,y_test_OneHot,verbose=2)
    print(scores)

