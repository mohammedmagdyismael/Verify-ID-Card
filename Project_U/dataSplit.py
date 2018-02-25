# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:54:03 2018

@author: mohammed-PC
"""

import cv2
import numpy as np

#Input : RGB Image(Element of the Returned values of the getMainROIs method)
#Return: List of RGB Images
def dataSplit(image_slice):
    newview1 = image_slice
    copy1 = image_slice
    newview1 = cv2.fastNlMeansDenoisingColored(newview1,None,10,10,7,21)
    newview1 = cv2.cvtColor(newview1, cv2.COLOR_BGR2GRAY)
    hist,bins = np.histogram(newview1.flatten(),256,[0,256])
    hist.cumsum()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    newview1 = clahe.apply(newview1)
    low = np.array([0])
    upper = np.array([55])
    newview1 = cv2.inRange(newview1 ,low , upper )
    kernel = np.ones((33,33), np.uint8)
    newview1 = cv2.dilate(newview1, kernel, iterations=1)
    newview1, contours1, hierarchy = cv2.findContours(newview1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    lst1 = []
    slices1 = []

    for i in range(0,len(contours1)):
        arr = [x,y,w,h] = cv2.boundingRect(contours1[i])
        x,y,w,h = cv2.boundingRect(contours1[i])
        lst1.append(arr)
        y0 = lst1[i][1]
        x0 = lst1[i][0]
        yf0 = (lst1[i][1] + lst1[i][3] )
        xf0 = (lst1[i][0] + lst1[i][2] )
        slices1.append(copy1 [y0:yf0 , x0:xf0 ])
    return slices1
