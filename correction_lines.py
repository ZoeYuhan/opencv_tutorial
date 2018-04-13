#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/10 19:34
# @Author  : Zoe
# @Site    : 
# @File    : correction_lines.py
# @Software: PyCharm Community Edition
import cv2
import numpy as np
from matplotlib import pyplot as plt


# 度数转换
def DegreeTrans(theta):
    return theta/np.pi * 180


# 逆时针旋转图像degree角度（原尺寸）
def rotateImage(img, degree):
    rows, cols = img.shape[0], img.shape[1]
    # 旋转中心为图像中心
    center_x = float(cols / 2.0)
    center_y = float(rows / 2.0)
    center = (center_x, center_y)
    length = int(np.sqrt(cols * cols + rows * rows))

    # 计算二维旋转的仿射变换矩阵
    M = cv2.getRotationMatrix2D(center, degree, 1)
    # 仿射变换，背景色填充为白色
    img_rotate = cv2.warpAffine(img, M, (length, length), 1, cv2.BORDER_CONSTANT, 1)
    return img_rotate


def CalcDegree(img):
    midImage = cv2.Canny(img, 50, 200, 3)
    dstImage = cv2.cvtColor(midImage,  cv2.COLOR_GRAY2BGR)
    # 通过霍夫变换检测直线
    lines = cv2.HoughLines(midImage,  1, np.pi / 180, 300, 0, 0)  # 第5个参数就是阈值，阈值越大，检测精度越高

    if lines is None:
        lines = cv2.HoughLines(midImage,  1, np.pi / 180, 200, 0, 0)

    if lines is None:
        lines = cv2.HoughLines(midImage, 1, np.pi / 180, 150, 0, 0)

    if lines is None:
        print("没有检测到直线！")
        return None
    #  依次画出每条线段
    sum = 0
    for rho, theta in lines[:, 0, :]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        # 只选角度最小的作为旋转角度
        sum += theta

        cv2.line(dstImage, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
    plt.imshow(dstImage)
    plt.title('detect lines')
    plt.show()
    average = sum / len(lines)

    angle = DegreeTrans(average) - 90
    print(sum, lines.shape, average, DegreeTrans(average))
    dst = rotateImage(img, angle)
    plt.imshow(dst)
    plt.title('Rotated img')
    plt.show()
    return dst, angle


def ImageRecify(img):
    # 倾斜角度矫正
    dst, degree = CalcDegree(img)



if __name__ == '__main__':

    # image = cv2.imread('./data/1.jpg', 0)
    image = cv2.imread('./data/8.jpg')
    if image is None:
        raise Exception("No image load!")

    dst, degree = CalcDegree(image)
    print(degree)
