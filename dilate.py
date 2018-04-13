#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/9 15:20
# @Author  : Zoe
# @Site    : 
# @File    : dilate.py
# @Software: PyCharm Community Edition
import cv2
import numpy as np
from matplotlib import pyplot as plt


def dilate(img):
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=1)
    titles = ['Original Image', 'Dilate Image']
    images = [img, dilate]
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
    dilate(image)

