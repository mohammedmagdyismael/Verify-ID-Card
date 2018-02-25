# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:13:12 2018

@author: mohammed-PC
"""

import cv2
import numpy as np

#Get Word Line Using Hough Line Detector
#Input: Processed Image, Word in Black and White background
#Return: Image of the word with the line drawn AND the Index of the row where the line is found
def getWordLine1(image):
    h , w , l = image.shape
    #convert Input image RGB -> Gray
    gry = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #Get Edges using Canny Edge Detector
    edges = cv2.Canny(gry,50,150,apertureSize = 3)
    #Get Word Line Using Hough Line Detector
    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        #x0 = a*rho
        y0 = b*rho
        #x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        #x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    #Select Pixels Row
    rowIndex = int((y1+y2)/2.0)
    row_of_pixels = image [rowIndex:rowIndex , 0:w-1 ]
    #Invert the Row
    row_of_pixels = cv2.bitwise_not(row_of_pixels)
    #Convert the row to gray scale so it can be thresholded to two values 0/255
    row_of_pixels = cv2.cvtColor(row_of_pixels , cv2.COLOR_BGR2GRAY)
    #loop on each pixel in the row to define its value even 0 or 255
    for i in range(0,len(row_of_pixels[0])):
        temp = row_of_pixels [0][i]
        if temp < 128:
            row_of_pixels [0][i] = 0
        elif temp > 128:
            row_of_pixels [0][i] = 255
    pixelsRow = row_of_pixels[0]

    return pixelsRow  , rowIndex
