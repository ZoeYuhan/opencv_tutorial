#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/9 16:30
# @Author  : Zoe
# @Site    : 
# @File    : Hough.py
# @Software: PyCharm Community Edition
"""
霍夫变换： 检测直线和圆

HoughLines：  标准霍夫变换  和  多尺度霍夫线性变换
HoughLinesP： 累积概率霍夫线性变换
HoughCircles： 
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

def Houghline(img):
    src_img = copy.copy(img)
    # 滤波
    img_filted = cv2.GaussianBlur(img, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(img_filted, 50, 200, apertureSize=3)

    # 直线检测
    lines = cv2.HoughLines(edges, 1, np.pi/180, 1050, 0, 0)
    print(lines.shape)
    for rho, theta in lines[:, 0, :]:

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*a)
        print(rho, theta)
        print((x1, y1), (x2, y2))
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_AA)

    images = [src_img,  img_filted,
              src_img,  edges,
              src_img,  img]
    titles = ['Original Image', "Filter",
              'Original Image', "Canny detect",
              'Original Image', "Houghline"]

    for i in range(3):
        plt.subplot(3, 2, i * 2 + 1), plt.imshow(images[i * 2])
        plt.title(titles[i * 2]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 2, i * 2 + 2), plt.imshow(images[i * 2+1])
        plt.title(titles[i * 2 + 1]), plt.xticks([]), plt.yticks([])

    plt.show()


def HoughlinesP(img):
    src_img = copy.copy(img)
    # 滤波
    img_filted = cv2.GaussianBlur(img, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(img_filted, 50, 200, apertureSize=3)

    # 直线检测
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 500, minLineLength, maxLineGap)
    print(lines.shape)
    for x1, y1, x2, y2 in lines[:, 0, :]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    images = [src_img, img_filted,
              src_img, edges,
              src_img, img]
    titles = ['Original Image', "Filter",
              'Original Image', "Canny detect",
              'Original Image', "HoughlinesP"]

    for i in range(3):
        plt.subplot(3, 2, i * 2 + 1), plt.imshow(images[i * 2])
        plt.title(titles[i * 2]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 2, i * 2 + 2), plt.imshow(images[i * 2 + 1])
        plt.title(titles[i * 2 + 1]), plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == '__main__':

    # image = cv2.imread('./data/1.jpg', 0)
    image = cv2.imread('./data/3.jpg')
    if image is None:
        raise Exception("No image load!")
    HoughlinesP(image)
