#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/31 13:46
# @Author  : 陈伟峰
# @Site    : 
# @File    : 自编码器异常检测.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection as skm
from sklearn.metrics import accuracy_score, precision_score,recall_score


ECG = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv',header=None)
print(ECG.head())

print(ECG.shape)

ECG5000_data = ECG.values
ECG_label = ECG5000_data[:, -1]   #读取标签
ECG_data = ECG5000_data[:,0:-1]  #读取数据

train_data, test_data, train_label, test_label = skm.train_test_split(ECG_data, ECG_label,test_size=0.3)

min_data = tf.reduce_min(train_data)
max_data = tf.reduce_max(train_data)
train_data = (train_data - min_data) / (max_data - min_data)
test_data = (test_data - min_data) / (max_data - min_data)
train_data = tf.cast(train_data, tf.float64)
test_data = tf.cast(test_data, tf.float64)


