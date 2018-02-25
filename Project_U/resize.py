# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:21:27 2018

@author: mohammed-PC
"""

import cv2
import numpy as np

#fit the Image size to the height and width of the statment included
#Input: Image
#Return: new dimensions to be used in resizing
def resize (originaImage):
    colsSum = []
    colsSpaces = []
    h , w , l = originaImage.shape
    #invert it
    inv = cv2.bitwise_not(originaImage)
    inv = cv2.cvtColor(inv ,  cv2.COLOR_BGR2GRAY)
    low = np.array([128])
    upper = np.array([255])
    inv = cv2.inRange(inv ,low , upper )
    #get sum of columns from dilated-Inverted image
    for c in range(0,w):
        col = inv [0:h , c:c+1 ]
        x = np.count_nonzero (col)
        colsSum.append(x)
    rowsSum = []
    for r in range(0,h):
        row = inv [r:r+1 , 0:w ]
        x = np.count_nonzero (row)
        rowsSum.append(x)

    flagL = 0
    for i in range(0,len(colsSum)):
         if colsSum[i] != 0 and flagL == 0:
            alignL = i
            flagL = 1

    flagR = 0
    for i in range(len(colsSum)-1 , -1 , -1):
         if colsSum[i] != 0 and flagR == 0:
            alignR = i
            flagR = 1

    flagU = 0
    for i in range(0,len(rowsSum)):
         if rowsSum[i] != 0 and flagU == 0:
            alignU = i
            flagU = 1

    flagD = 0
    for i in range(len(rowsSum)-1 , -1 , -1):
         if rowsSum[i] != 0 and flagD == 0:
            alignD = i
            flagD = 1
    #
    return alignR , alignL , alignD , alignU