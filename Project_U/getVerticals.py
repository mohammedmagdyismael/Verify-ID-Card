# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:15:57 2018

@author: mohammed-PC
"""
import cv2
import numpy as np 
from matplotlib import pyplot as plt

#Get the summations of non zero elements in each column (contains word) and zero elements in each column (spaces)
#Input: Processed Image, Word in Black and White background
#Return: sum of non zeros elements in each column (colsSum) and zero columns which are spaces (colsSpaces)
def getVerticals(image, kernelsize):
    colsSum = []
    colsSpaces = []
    h , w , l = image.shape
    #dilate the image
    kernel = np.ones(kernelsize, np.uint8)
    dil = cv2.dilate(image, kernel, iterations=1)
    #invert it
    inv = cv2.bitwise_not(dil)
    inv = cv2.cvtColor(inv ,  cv2.COLOR_BGR2GRAY) ##
    low = np.array([128])
    upper = np.array([255])
    inv = cv2.inRange(inv ,low , upper )
    plt.figure()
    plt.imshow(inv)
    #get sum of columns from dilated-Inverted image
    for c in range(0,w):
        col = inv [0:h , c:c+1 ]
        x = np.count_nonzero (col)
        colsSum.append(x)
        if x == 0 :
            colsSpaces.append(c)
    maxPeak = max(colsSum)
    return colsSum , colsSpaces ,maxPeak