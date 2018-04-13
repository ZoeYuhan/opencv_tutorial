#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/10 20:55
# @Author  : Zoe
# @Site    : 
# @File    : scanner.py
# @Software: PyCharm Community Edition
import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt


# 计算两条直线的交点
def ComputeIntersect(a, b):
    x1, y1, x2, y2 = a[0], a[1], a[2], a[3]
    x3, y3, x4, y4 = b[0], b[1], b[2], b[3]    
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d:
        x = ((x2 - x1) * (x4 - x3) * (y1 - y3) - x3 * (x2 - x1) * (y3 - y4) + x1 * (y2 - y1) * (x3 - x4))/  ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        y = ((y2 - y1) * (y4 - y3) * (x1 - x3) - y3 * (y2 - y1) * (x3 - x4) + y1 * (x2 - x1) * (y3 - y4)) /  ((x4 - x3) * (y2 - y1) - (y4 - y3) * (x2 - x1))
        return x, y
    else:
        return -1, -1


# 两条直线距离太近
def IsBadLine(a, b):
    if abs(a) + abs(b) < 100:
        return True
    else:
        return False

# 确定四个点的中心线
def SortCorners(corners, center):
    backup = copy.copy(corners)
    corners_arg = np.argsort(corners[:,0,0])
    corners = corners[corners_arg]
    top = []
    bot = []

    for i in range(len(corners)):
        if corners[i, 0, 1] < center[0][1] and len(top) < 2:
            top.append(corners[i, 0, :])
        else:
            bot.append(corners[i, 0, :])
    corners = []
    if len(top) == 2 and len(bot) == 2:
        tl = top[1] if top[0][0] > top[1][0] else top[0]
        tr = top[0] if top[0][0] > top[1][0] else top[1]
        bl = bot[1] if bot[0][0] > bot[1][0] else bot[0]
        br = bot[0] if bot[0][0] > bot[1][0] else bot[1]

        corners.append([tr,tl, br, bl])
    else:
        corners = backup
    return corners


def CalcDstSize(corners):
    corners = corners[0]
    h1 = np.sqrt((corners[0][0] - corners[3][0]) * (corners[0][0] - corners[3][0]) + (corners[0][1] - corners[3][1]) * (
    corners[0][1] - corners[3][1]))
    h2 = np.sqrt((corners[1][0] - corners[2][0]) * (corners[1][0] - corners[2][0]) + (corners[1][1] - corners[2][1]) * (
    corners[1][1] - corners[2][1]))
    g_dst_hight = max(h1, h2)

    w1 = np.sqrt((corners[0][0] - corners[1][0]) * (corners[0][0] - corners[1][0]) + (corners[0][1] - corners[1][1]) * (
    corners[0][1] - corners[1][1]))
    w2 = np.sqrt((corners[2][0] - corners[3][0]) * (corners[2][0] - corners[3][0]) + (corners[2][1] - corners[3][1]) * (
    corners[2][1] - corners[3][1]))
    g_dst_width = max(w1, w2)

    return g_dst_hight, g_dst_width


def warpPerspective(image):
    # 灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    img_filter = cv2.GaussianBlur(gray, (5, 5), 0)

    # 获取自定义核
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 膨胀操作
    img_dilate = cv2.dilate(img_filter, element)

    # 边缘提取
    edges = cv2.Canny(img_dilate, 30, 120, 3)
    plt.imshow(edges)
    plt.title("Edges")
    plt.show()
    #  CV_RETR_EXTERNAL，只检索外框
    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找轮廓

    # 找到面积最大的轮廓
    max_area = 0
    index = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            max_area = area
            index = i

    contours = contours[index]

    for line_type in range(1, 3):
        black = np.zeros(edges.shape, np.uint8)

        # 画边框
        cv2.drawContours(black, contours, -1, (255, 0, 0), line_type)
        plt.imshow(black)
        plt.title('black with contours with line_type: %s' % str(line_type))
        plt.show()
        flag = 0
        for para in range(1, 300):
            lines = cv2.HoughLinesP(black, 1, np.pi / 180, para, 30, 10)

            if lines is None:
                continue

            # 过滤距离太近的直线
            ErasePt = []
            for i in range(lines.shape[0]):
                for j in range(i + 1, lines.shape[0]):
                    if (IsBadLine(abs(lines[i, 0, 0] - lines[j, 0, 0]), abs(lines[i, 0, 1] - lines[j, 0, 1])) and (
                            IsBadLine(abs(lines[i, 0, 2] - lines[j, 0, 2]), abs(lines[i, 0, 3] - lines[j, 0, 3])))):
                        ErasePt.append(j)

            lines = np.delete(lines, ErasePt, 0)

            if lines.shape[0] != 4:
                continue
            a = np.zeros(black.shape, np.int8)
            for x1, y1, x2, y2 in lines[:, 0, :]:
                cv2.line(a, (x1, y1), (x2, y2), (255, 0, 0), line_type, cv2.LINE_AA)

            plt.imshow(a)
            plt.title('The lines with para: %s' % str(para))
            plt.show()
            # 计算直线的交点，保存在图像范围内的部分
            corners = []
            for i in range(lines.shape[0] - 1):
                for j in range(i + 1, lines.shape[0]):
                    pt = ComputeIntersect(list(lines[i, 0, :]), list(lines[j, 0, :]))
                    if (0 <= pt[0] <= edges.shape[1]) and (0 <= pt[1] <= edges.shape[0]):
                        corners.append(pt)

            if len(corners) != 4:
                continue

            IsGoodPoints = True

            # 保证点与点的距离足够大 从而排除错误点
            for i in range(len(corners) - 1):
                for j in range(i + 1, len(corners)):
                    distance = np.sqrt((corners[i][0] - corners[j][0]) * (corners[i][0] - corners[j][0]) +
                                       (corners[i][1] - corners[j][1]) * (corners[i][1] - corners[j][1]))
                    if distance < 5:
                        IsGoodPoints = False
            if not IsGoodPoints:
                continue

            # 轮廓近似
            corners = np.array(corners, np.int)

            corners = corners.reshape(corners.shape[0], 1, corners.shape[1])
            perimeter = cv2.arcLength(corners, True)
            approx = cv2.approxPolyDP(corners, perimeter * 0.02, True)
            if lines.shape[0] == 4 and len(corners) == 4 and len(approx) == 4:
                flag = 1
                break
        # Get mass center
        center = np.zeros((1, 2))

        for i in range(len(corners)):
            center += corners[i]

        center *= (1. / len(corners))

        if flag:
            print("We found it! The corner is: %s " % str(corners))
            a = np.zeros(black.shape, np.int8)
            for x1, y1, x2, y2 in lines[:, 0, :]:
                cv2.line(a, (x1, y1), (x2, y2), (255, 0, 0), line_type, cv2.LINE_AA)

            plt.subplot(1, 2, 1)
            plt.imshow(a)
            plt.title('The lines with para: %s' % str(para))

            bkup = copy.copy(image)
            bkup = cv2.circle(bkup, tuple(corners[0, 0, :]), 3, (255, 0, 0), -1)
            bkup = cv2.circle(bkup, tuple(corners[1, 0, :]), 3, (0, 255, 0), -1)
            bkup = cv2.circle(bkup, tuple(corners[2, 0, :]), 3, (0, 0, 255), -1)
            bkup = cv2.circle(bkup, tuple(corners[3, 0, :]), 3, (255, 255, 255), -1)
            bkup = cv2.circle(bkup, (int(center[0][0]), int(center[0][1])), 3, (255, 0, 255), -1)
            plt.subplot(1, 2, 2)
            plt.imshow(bkup)
            plt.title('The corners')
            plt.show()

            corners = SortCorners(corners, center)
            g_dst_width, g_dst_hight = CalcDstSize(corners)
            quad = np.zeros((g_dst_hight, g_dst_width))
            quad_pts = list()
            quad_pts.append((0, 0))
            quad_pts.append((quad.shape[0], 0))
            quad_pts.append((0, quad.shape[1]))
            quad_pts.append((quad.shape[0], quad.shape[1]))

            # 透视变换
            transmtx = cv2.getPerspectiveTransform(np.array(corners, np.float32), np.array(quad_pts, np.float32))
            print(transmtx)
            quad = cv2.warpPerspective(image, transmtx, quad.shape)
            quad = cv2.flip(quad, 1)
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("original image")
            plt.subplot(1, 2, 2)
            plt.imshow(quad)
            plt.title('quadrilateral')
            plt.show()
            return 0

if __name__ == '__main__':
    image = cv2.imread("./data/7.png")
    warpPerspective(image)

    from collections import Counter

#for para in range(10,10,300):
#    print(para)
#    lines = cv2.HoughLinesP(edges, 1, np.pi/180, para, 30, 10)
#    a = np.zeros(black.shape,np.int8)
#    
#    for x1,y1,x2,y2 in lines[:,0,:]:
#        cv2.line(a, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_AA)
#        
#    plt.imshow(a)
#    plt.show()