#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 11:43
# @Author  : 陈伟峰
# @Site    : 
# @File    : 张量的基本运算.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf


# 常量
scalar_int = tf.constant(2)
scalar_float = tf.constant(3.14)
scalar_str = tf.constant(["hello","world"])

print(scalar_str)
print(scalar_float)
print(scalar_int)

matrix_int = tf.constant([[1,2],[2,3]])
matrix_float = tf.constant([[1.,2.],[2.,3.]])
matrix_str =  tf.constant([['hello'],['world']])
print(matrix_int.shape)
print(matrix_float.shape)
print(matrix_str.shape)


li = [1,2,3]   #创建python列表
array = np.array([1.0,2.0,3.0])   #创建numpy数组
print(tf.convert_to_tensor(li))   #将列表转换为tensor
print(tf.convert_to_tensor(array))   #将numpy数组转换为tensor


print(tf.zeros((3,3),dtype=tf.int8))   #创建3x3的全0矩阵
print(tf.ones((3,3)))   #创建3x3的全1矩阵


print(tf.fill((5,5),value=2))

print(tf.random.normal((3,3),2,4))   #创建正态分布张量

print(tf.random.uniform((3,3),0,10))   #创建均匀分布张量

print(tf.add(2,3))   #加法
print(tf.subtract(2,3))   #减法
print(tf.multiply(2,3))   #乘法
print(tf.divide(2,3))   #除法

a = tf.constant(2)
b = tf.constant(3)
print(a+b)   #加法
print(a-b)   #减法
print(a*b)   #乘法
print(a/b)   #除法

print(tf.abs(-2))    #求绝对值
print(tf.pow(2,2))   #求次方
float_number = tf.cast(4,tf.float32)   #强制转换类型
print(tf.sqrt(float_number))   #开平方

a = tf.random.normal([5,6])   #创建5x6正态分布张量
b = tf.random.normal([6,5])   #创建6x5正态分布张量
print(tf.matmul(a,b))   #矩阵相乘

print(a@b)   #矩阵相乘


a = tf.random.uniform((5,5),2,10)
print(a)
print(a.shape)


print(a[1])
print(a[2][2])
print(a[2,0:2])

a = tf.random.normal((3,4))
print(a)

print(tf.reshape(a,(2,6)))
print(tf.reshape(a,(2,-1)))


a = tf.random.normal((28,28))
a = tf.expand_dims(a,axis=0)   #增添第一维
print(a.shape)
a = tf.expand_dims(a,axis=3)   #增添第四维
print(a)
print(a.shape)


a = tf.random.normal((1,28,28,1))
a = tf.squeeze(a,axis=0)#删除第一个维度
print(a.shape)
a = tf.squeeze(a,axis=2)   #删除最后一个维度
print(a.shape)

a = tf.random.normal((1,1,28,28))
a = tf.transpose(a,[0,2,3,1])
print(a.shape)


a = tf.constant(6)
b = tf.constant(6.0)
print(tf.strings.as_string(a))
print(tf.strings.as_string(b,precision=1))


a = tf.constant(['hello'])
print(tf.strings.bytes_split(a))


a = tf.constant(['h','e','l','l','o'])
print(tf.strings.join(a))   #字符拼接


a = tf.constant("ABCD")
print(tf.strings.lower(a))   #小写转换

a = tf.constant("abcd")
print(tf.strings.upper(a))   #大写转换



a = tf.ragged.constant([[1,2,3],[],[4,5,6]])   #创建整型Ragged张量
b = tf.ragged.constant([['hello'],[],['world']])   #创建字符串型Ragged张量
print(a)
print(b)

# c = tf.ragged.constant([[1,2,3],[],['hello']])

a = tf.ragged.constant([[1,2,3],[],[4,5,6]])
print(a+2)
print(a-2)
print(a*2)
print(a/2)


a = tf.ragged.constant([[1,2,3],[],[4,5,6]])
rule = lambda x: x * 3 + 1
print(tf.ragged.map_flat_values(rule,a))   #函数映射变换
