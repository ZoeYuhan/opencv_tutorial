#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/9 15:16
# @Author  : Zoe
# @Site    : 
# @File    : erode.py
# @Software: PyCharm Community Edition
import cv2
import numpy as np
from matplotlib import pyplot as plt


def erode(img):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    titles = ['Original Image', 'Erosion Image']
    images = [img, erosion]
    plt.figure()
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    # 读入灰度图像
    image = cv2.imread('./data/1.jpg', 0)
    if image is None:
        raise Exception("No image load!")
    erode(image)

