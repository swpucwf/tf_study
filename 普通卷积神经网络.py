#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 21:27
# @Author  : 陈伟峰
# @Site    : 
# @File    : 普通卷积神经网络.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

# 分别读入cifar-10的数据集的训练集数据和测试集数据
(x_train_image, y_train_label), (x_test_image, y_test_label) = tf.keras.datasets.cifar10.load_data()
print(x_train_image.shape)  # 输出训练集数据大小
print(y_train_label.shape)  # 输出训练集数据标签大小
print(x_test_image.shape)  # 输出测试集数据大小
print(y_test_label.shape)  # 输出测试集数据标签大小


def plot_image(images, labels, prediction, index, nums=10):
    fig = plt.gcf()
    fig.set_size_inches(14, 14)  # 设置图表大小
    for i in range(0, nums):
        ax = plt.subplot(5, 5, 1 + i)  # 子图生成
        ax.imshow(images[index])  # index是为了方便索引所要查询的图像
        title = 'label:' + str(labels[index][0])  # 定义title方便图像结果对应
        if len(prediction) > 0:  # 如果有预测图像，则显示预测结果
            title += 'prediction:' + str(prediction[index])
        ax.set_title(title, fontsize=13)  # 设置图像title
        ax.set_xticks([])  # 无x刻度
        ax.set_yticks([])  # 无y刻度
        index += 1


plot_image(x_train_image, y_train_label, [], 0, 10)

x_train_normalize = x_train_image.astype('float32') / 255  # 训练集归一化
x_test_normalize = x_test_image.astype('float32') / 255  # 测试集归一化
y_train_OneHot = tf.keras.utils.to_categorical(y_train_label)  # one-hot编码
y_test_OneHot = tf.keras.utils.to_categorical(y_test_label)

print(x_train_image[0][0][0])
print(x_train_normalize[0][0][0])

model = tf.keras.models.Sequential()  # 模型创建
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3),
                              activation='relu'))  # 卷积层
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # 最大池化层
model.add(keras.layers.Flatten())  # 将卷积输出打平成一维，方便与全连接层连接
model.add(keras.layers.Dense(1500, activation='relu'))  # 全连接层
model.add(keras.layers.Dropout(0.3))  # Dropout层，30%神经元失活
model.add(keras.layers.Dense(10, activation='softmax'))  # 输出10个类别

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_train = model.fit(x=x_train_normalize, y=y_train_OneHot, validation_split=0.2, epochs=10, batch_size=300,
                        verbose=1)

print(model_train.history)

def train_history(model_train, train, val):  # 训练集和验证集的准确率变化曲线
    plt.plot(model_train.history[train])
    plt.plot(model_train.history[val])
    plt.title('Train History')
    plt.xlabel('epoch')  # 训练次数
    plt.ylabel(train)
    plt.legend(['train', 'validation'], loc='upper left')  # 图例


train_history(model_train, 'accuracy', 'val_accuracy')
train_history(model_train, 'loss', 'val_loss')

scores = model.evaluate(x_test_normalize, y_test_OneHot, verbose=2)  # 测试集测试
prediction = model.predict_classes(x_test_normalize)  # 类别预测
print(prediction)

plot_image(x_test_image, y_test_label, prediction, 0, 10)

prediction_probability = model.predict(x_test_normalize)  # 概率预测
print(prediction_probability[0])

# 数字与对应的类别
label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
              8: 'ship', 9: 'truck'}


def predicted_probability(x, y, prediction, prediction_probability, i):  # 显示图像不同类别的预测概率
    plt.figure(figsize=(2, 2))
    plt.imshow(x[i])
    plt.show()
    print("label:", label_dict[y[i][0]], 'predict:', label_dict[prediction[i]])  # 预测结果和真实标签
    for j in range(10):  # 输出10个类别概率
        print(label_dict[j] + 'Probability:%1.9f' % (prediction_probability[i][j]))


predicted_probability(x_test_image, y_test_label, prediction, prediction_probability, 0)

pd.crosstab(y_test_label.reshape(-1), prediction, rownames=['label'], colnames=['prediction'])
