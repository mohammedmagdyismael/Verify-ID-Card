# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:10:09 2018

@author: mohammed-PC
"""
import cv2

#Get Row of pixels where the Word line is found
#Input: Processed Image, Word in Black and White background
#Return: List of Pixels of the Word Line Row
def getPixelsRow(rawimage , rowIndex):
    h , w , l = rawimage.shape
    #select the row
    row_of_pixels = rawimage [rowIndex:rowIndex + 1 , 0:w ]
    #Invert the row
    row_of_pixels = cv2.bitwise_not(row_of_pixels)
    #Convert the row to gray scale so it can be thresholded to two values 0/255
    row_of_pixels = cv2.cvtColor(row_of_pixels , cv2.COLOR_BGR2GRAY) ##
    #loop on each pixel in the row to define its value even 0 or 255
    for i in range(0,len(row_of_pixels[0])):
        temp = row_of_pixels [0][i]
        if temp < 128:
            row_of_pixels [0][i] = 0
        elif temp > 128:
            row_of_pixels [0][i] = 255
    pixelsRow = row_of_pixels[0]

    return pixelsRow