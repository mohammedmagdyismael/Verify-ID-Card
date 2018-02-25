# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:50:42 2018

@author: mohammed-PC
"""
import cv2
import numpy as np

def bk_getMainROIs(image_file):
    im = image_file
    copy = im
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(im, (9, 9), 0)
    edges = cv2.Canny(im,140,200)
    edges = cv2.resize(edges, (900, 400))
    kernel = np.ones((40,40), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges, contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    lst = []
    slices = []
    newview = cv2.resize(copy, (900, 400))

    for i in range(0,len(contours)):
        arr = [x,y,w,h] = cv2.boundingRect(contours[i])
        x,y,w,h = cv2.boundingRect(contours[i])
        lst.append(arr)
        y0 = lst[i][1] - 3
        if y0 < 0:
            y0 = lst[i][1]
        x0 = lst[i][0]
        yf0 = (y0 + lst[i][3] )
        xf0 = (x0 + lst[i][2] )
        slices.append(newview [y0:yf0, x0:xf0 ])

    return slices    