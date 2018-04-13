#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/9 15:25
# @Author  : Zoe
# @Site    : 
# @File    : Filter.py
# @Software: PyCharm Community Edition
"""
Filter: 线性滤波 和 非线性滤波
        线性滤波： 方框滤波，均值滤波，高斯滤波   （速度相对快）
        非线性滤波： 中值滤波，双边滤波      （速度相对慢）
        
选择策略：
    1. 优先高斯滤波 然后均值滤波
    2. 斑点和椒盐噪声优先 中值滤波
    3. 去除噪声同时保留边缘信息 使用双边滤波
    
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def liner_filter(img):
    """
    线性滤波器
    :param img: 
    :return: 
    """
    # 方框滤波
    img_box = cv2.boxFilter(img, -1, (3, 3), normalize=True)

    # 均值滤波
    img_mean = cv2.blur(img, (5, 5))

    # 高斯滤波
    img_guassian = cv2.GaussianBlur(img, (5, 5), 0)

    images = [img, 0, img_box,
              img, 0, img_mean,
              img, 0, img_guassian]
    titles = ['Original Image', 'Filtered Histogram', 'Box Filter',
              'Original Image', 'Filtered Histogram', "Mean Filter",
              'Original Image', 'Filtered Histogram', "Guassian Filter"]

    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3 + 2].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()


def no_linear_filter(img):
    """
    非线性滤波器
    :param img: 
    :return: 
    """
    # 中值滤波
    img_median = cv2.medianBlur(img, 5)

    # 双边滤波
    # cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
    # d – Diameter of each pixel neighborhood that is used during filtering.
    # If it is non-positive, it is computed from sigmaSpace
    # 9 邻域直径，两个75 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差
    img_bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    images = [img, 0, img_median,
              img, 0, img_bilateral]
    titles = ['Original Image', 'Filtered Histogram', 'Median Filter',
              'Original Image', 'Filtered Histogram', "Bilateral Filter"]

    for i in range(2):
        plt.subplot(2, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 3, i * 3 + 2), plt.hist(images[i * 3 + 2].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    # 读入灰度图像
    image = cv2.imread('./data/1.jpg', 0)
    if image is None:
        raise Exception("No image load!")
    liner_filter(image)
    no_linear_filter(image)


