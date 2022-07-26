import os
import random

import cv2
import numpy as np


class LeftPost:
    def __init__(self):
        self.lower = np.array([0,0,0])
        self.upper = np.array([180,255,179])
        self.outside = []
        self.inside = []

    def line_fit(self, lines):
        loc = []  # 坐标
        for line in lines:
            x1, y1, x2, y2 = line[0]
            loc.append([x1, y1])
            loc.append([x2, y2])
        loc = np.array(loc)  # loc 必须为矩阵形式，且表示[x,y]坐标

        output = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)

        k = output[1] / output[0]
        b = output[3] - k * output[2]

        want_xia = (240 - b) / k
        want_shang = (0 - b) / k

        return np.array([[int(want_shang), 0, int(want_xia), 240]])

    def show_img(self, img, delay=33):
        cv2.imshow('img',img)
        cv2.waitKey(delay)

    def color_mask(self, image):
        frame = np.copy(image)
        frame = cv2.GaussianBlur(frame,(5,5),1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.bitwise_not(mask)
        mask = cv2.dilate(mask, (3,3), iterations=3)
        mask = cv2.erode(mask, (3,3), iterations=3)
        return mask

    def roi_mask(self, gray_img):
        """
        对gray_img进行掩膜
        :param gray_img: 灰度图,channels=1
        """
        poly_pts = np.array([[[114,240], [125,0], [187, 0], [187, 240]]])
        mask = np.zeros_like(gray_img)
        mask = cv2.fillPoly(mask, pts=poly_pts, color=255)
        img_mask = cv2.bitwise_and(gray_img, mask)
        return img_mask

    def calculate_slope(self, line):
        """
        计算线段line的斜率
        :param line: np.array([[x_1, y_1, x_2, y_2]])
        :return:
        """
        x_1, y_1, x_2, y_2 = line[0]
        return (y_2 - y_1) / (x_2 - x_1 + 1e-8)

    def cluster(self, lines):
        this = [lines[0]]
        another = []
        for i in range(1, len(lines)):
            slope = self.calculate_slope([lines[i][0]])
            if slope < -10:
                distance = abs((lines[0][0][0] - lines[i][0][0]))
                if distance < 3:
                    this.append(lines[i])
                else:
                    another.append(lines[i])
        return this,another

    def make_robust(self, one_list, num=15):
        if len(one_list) > num:
            one_list.pop(0)
        return one_list

    def get_lines(self, img):
        self.color_img = img.copy()
        img = self.color_mask(img)

        canny_img = cv2.Canny(img, 100, 150)
        canny_img = self.roi_mask(canny_img)

        many_lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 40, minLineLength=100,
                                     maxLineGap=50)

        this, another = self.cluster(many_lines)

        if another and this:
            this = self.line_fit(this)
            another = self.line_fit(another)

            if this[0][0] < another[0][0]:
                self.outside.append(this)
                self.inside.append(another)
            else:
                self.outside.append(another)
                self.inside.append(this)

        outside = self.make_robust(self.outside)
        inside = self.make_robust(self.inside)

        self.inner = self.line_fit(inside)
        self.outer = self.line_fit(outside)

        lines = [self.inner, self.outer]

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(self.color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)


if __name__ == '__main__':
    cap = cv2.VideoCapture('videos/left.avi')

    leftpost = LeftPost()
    while True:
        ret, img = cap.read()
        if not ret:
            break
        leftpost.get_lines(img)
        leftpost.show_img(leftpost.color_img)
        print(leftpost.inner)

