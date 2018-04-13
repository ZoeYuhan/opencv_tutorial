#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/10 15:00
# @Author  : Zoe
# @Site    : 
# @File    : correction_contours.py
# @Software: PyCharm Community Edition
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy


def correct_contours(img):
    src_img = copy.copy(img)

    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray)
    plt.title('gray')
    plt.show()
    # 二值化
    _, bin_img = cv2.threshold(gray,  100, 200, cv2.THRESH_BINARY)
    plt.imshow(bin_img)
    plt.title('thresh')
    plt.show()

    # 找轮廓边框
    _, contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        rectpoint = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3[0] 获取最小外接矩形的4个顶点
        angle = rect[2]
        print(angle)
        line1 = np.sqrt((rectpoint[1][1] - rectpoint[0][1]) * (rectpoint[1][1] - rectpoint[0][1]) + (
                         rectpoint[1][0] - rectpoint[0][0]) * (rectpoint[1][0] - rectpoint[0][0]))
        line2 = np.sqrt((rectpoint[3][1] - rectpoint[0][1]) * (rectpoint[3][1] - rectpoint[0][1]) + (
                         rectpoint[3][0] - rectpoint[0][0]) * (rectpoint[3][0] - rectpoint[0][0]))

        # 如果面积太小直接pass
        if line1 * line2 < 600:
            continue

        # 为了让正方形横着放，所以旋转角度是不一样的。竖放的，给他加90度，翻过来
        if line1 > line2:
            angle = 90 + angle

        # RoiSrcImg = np.zeros(img.shape, np.uint8)
        # cv2.drawContours(bin_img, contours, -1, 255)

        # RoiSrcImg = copy.copy(bin_img)
        # plt.imshow(bin_img)
        # plt.title('bin_img after draw contours')
        # plt.show()

        # 对RoiSrcImg进行旋转
        center = rect[0]
        # 计算旋转加缩放的变换矩阵
        M2 = cv2.getRotationMatrix2D(center, angle, 1)

        # 仿射变换
        RatationedImg = cv2.warpAffine(img, M2, (img.shape[1], img.shape[0]), 1, cv2.BORDER_CONSTANT, 1)

        plt.imshow(RatationedImg)
        plt.title('after rotate image')
        plt.show()

    # 对ROI区域进行抠图

    # 对旋转后的图片进行轮廓提取
    for j in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[j])
        print(x, y, w, h)
        if w*h < 600:
            continue

        dst_img = RatationedImg[x:x+h, y:y+w, :]
    plt.imshow(dst_img)
    plt.title('RIO')
    plt.show()

if __name__ == '__main__':

    # image = cv2.imread('./data/1.jpg', 0)
    image = cv2.imread('./data/4.jpg')
    if image is None:
        raise Exception("No image load!")

    correct_contours(image)
