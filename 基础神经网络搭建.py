#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 19:19
# @Author  : 陈伟峰
# @Site    : 
# @File    : 基础神经网络搭建.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
np.random.seed(43)   #定义随机数生成种子
df = pd.DataFrame({   #创建DataFrame格式的数据
    'color':['black']*5+['white']*5,
    'age':np.random.rand(10)*10,
    'weight':np.random.rand(10)*100,
    'type':['cat']*5+['dog']*5
})

# df = df.join(pd.get_dummies(df.color))  #利用pandas调用one-hot编码函数
# df = df.join(pd.get_dummies(df['type']))
# print(df[0:10])

df.color[df.color=='black'] = 0
df.color[df.color=='white'] = 1

df.type[df.type=='cat'] = 0
df.type[df.type=='dog'] = 1
print(df[0:10])




x = df[['color','age','weight']].values.astype(np.float)
y = df[['type']].values.astype(np.int)

#
# model = keras.Sequential()   #创建Sequential模型
# model.add(keras.layers.Dense(50,input_dim = x.shape[1],activation='relu'))   #输入层
# model.add(keras.layers.Dense(25,activation='relu'))   #隐藏层
# model.add(keras.layers.Dense(y.shape[1],activation='sigmoid'))   #输出层
# print(model.summary())   #查看模型概况

model = keras.Sequential([
    keras.layers.Dense(50,input_dim = x.shape[1],activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(25,activation='relu'),
    keras.layers.Dense(y.shape[1],activation='sigmoid')
])
print(model.summary())   #查看模型概况
model.compile(loss='binary_crossentropy',optimizer='SGD')   #配置训练参数
# model.compile(loss='binary_crossentropy',optimizer='Adam')

x_test =  pd.DataFrame({   #单个测试数据
    'color':['black'],
    'age':np.random.rand(1)*10,
    'weight':np.random.rand(1)*100,
})
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)   #设置早停法参数

import os

checkpoint_path = "training_model/cp.ckpt"   #模型保存路径
checkpoint_dir = os.path.dirname(checkpoint_path)
callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_best_only=True,save_weights_only=True,verbose=1)
print(checkpoint_dir)
model.fit(x,y,epochs=500,validation_data=(x,y),callbacks=[callback])   #模型训练
x_test['color'] = pd.get_dummies(['color'])   #对color属性进行one-hot编码

print(x_test)
model.predict(x_test.values)   #模型预测
# 模型保存方式1
model.load_weights(checkpoint_path)
print("加载成功")
model.save("./model")
# 加载方式2
model = tf.keras.models.load_model("./model")
print(model)