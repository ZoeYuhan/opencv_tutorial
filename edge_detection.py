#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/9 15:54
# @Author  : Zoe
# @Site    : 
# @File    : edge_detection.py
# @Software: PyCharm Community Edition
"""
边缘检测步骤：
    step1: 滤波，消除噪音
    step2: 增强，使得边界轮廓更加明显
    step3: 选出边缘点
    
三种算法: Canny / Sobel / Laplacian
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def Canny(img):
    # 高斯滤波
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Canny 边缘检测
    edges = cv2.Canny(img, 100, 200)
    # plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    return edges


def Sobel(img):
    # 高斯滤波
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Sobel 算子
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    absX = cv2.convertScaleAbs(gx)  # 转回uint8
    absY = cv2.convertScaleAbs(gy)
    edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    return edges


def Laplacian(img):
    # 高斯滤波
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # 拉式算子
    gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    edges = cv2.convertScaleAbs(gray_lap)
    return edges

if __name__ == '__main__':
    # 读入灰度图像
    image = cv2.imread('./data/3.jpg', 0)
    if image is None:
        raise Exception("No image load!")
    canny_edges = Canny(image)
    sobel_edges = Sobel(image)
    laplacian_edges = Laplacian(image)

    # 画图
    images = [image,  canny_edges,
              image,  sobel_edges,
              image,  laplacian_edges]
    titles = ['Original Image', 'canny edges',
              'Original Image', "sobel edges",
              'Original Image', "laplacian edges"]

    for i in range(3):
        plt.subplot(3, 2, i * 2 + 1), plt.imshow(images[i * 2], 'gray')
        plt.title(titles[i * 2]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 2, i * 2 + 2), plt.imshow(images[i * 2 + 1], 'gray')
        plt.title(titles[i * 2 + 1]), plt.xticks([]), plt.yticks([])

    plt.show()

