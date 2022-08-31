#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/30 11:34
# @Author  : 陈伟峰
# @Site    : 
# @File    : hog特征提取.py
# @Software: PyCharm

from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img=cv2.imread('img.png',cv2.IMREAD_GRAYSCALE)   #读取图像
    #hog特征检测
    _, hog_img = hog(img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(8, 8),visualize=True)
    plt.imshow(hog_img,cmap ='gray')
    plt.show()