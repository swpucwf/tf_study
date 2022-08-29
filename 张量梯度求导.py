#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 15:48
# @Author  : 陈伟峰
# @Site    : 
# @File    : 张量梯度求导.py
# @Software: PyCharm
import tensorflow as tf

if __name__ == '__main__':
    a = tf.Variable(2)  # 创建标量张量
    b = tf.Variable([2])  # 创建向量张量
    c = tf.Variable([[2]])  # 创建矩阵张量
    print(a)
    print(b)
    print(c)

    a = tf.Variable(2)
    b = tf.Variable(3, trainable=False)
    print(a.trainable)  # 查看张量状态
    print(b.trainable)

    a = tf.Variable(2)
    print(a.assign(3))

    # a = tf.Variable(2)
    # print(a.assign('3'))

    x = tf.constant(3.0)
    with tf.GradientTape() as tape:  # 梯度求导
        tape.watch(x)  # 常量张量跟踪
        y = x * x * x
    dy_dx = tape.gradient(y, x)
    print(dy_dx)

    x = tf.Variable(3.0)
    with tf.GradientTape() as tape:  # 梯度求导
        tape.watch(x)  # 常量张量跟踪
        y = x * x * x
    dy_dx = tape.gradient(y, x)
    print(dy_dx)

    # x = tf.Variable(3.0,trainable=False)
    # with tf.GradientTape() as tape:
    #     y = x*x*x
    # dy_dx = tape.gradient(y,x)
    # d2y_d2x = tape.gradient(dy_dx,x)
    # print(d2y_d2x)

    x = tf.Variable(3.0)
    with tf.GradientTape() as tape:  # 二次梯度求导
        with tf.GradientTape() as tape2:
            y = x * x * x
        dy_dx = tape2.gradient(y, x)
    d2y_d2x = tape.gradient(dy_dx, x)
    print(dy_dx)
    print(d2y_d2x)

    x = tf.Variable(2.0)
    k = tf.Variable(3.0)
    # watch_accessed_variables 自动跟踪 persistent，只调用一次
    #
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x * x * x
        z = x * x
    dy_dx = tape.gradient(y, x)
    dz_dx = tape.gradient(z, x)
    print(1)
    print(dy_dx)
    print(dz_dx)
    del tape


    @tf.function   #将python函数转为静态图执行
    def add(x):
        return x + 1
    print(isinstance(add.get_concrete_function(1).graph, tf.Graph))  #判断是否为图

    # def add(x):
    #     return x + 1
    # print(isinstance(add.get_concrete_function(1).graph, tf.Graph))  #判断是否为图
    @tf.function
    def f(x):
        if x > 0:
            return x
        else:
            return -x
    print(f(tf.constant(-2)))


    array = []
    @tf.function
    def f(x):
        for i in range(len(x)):
            array.append(x[i])
        print(array)
    print(f(tf.constant([1, 2, 3])))


    @tf.function
    def f(x):
        array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        for i in range(len(x)):
            array = array.write(i,x[i])
        tf.print(array.stack())
    print(f(tf.constant([1, 2, 3])))

    @tf.function
    def f(x):
        return x + x
    f1 = f.get_concrete_function(tf.constant(1))
    f2 = f.get_concrete_function(tf.constant(2))
    print(f1 is f2)