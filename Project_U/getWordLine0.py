# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:10:59 2018

@author: mohammed-PC
"""
import cv2
import numpy as np

#Get Word Line Using Horizontal Projection
#Input: Processed Image, Word in Black and White background
#Return: Image of the word with the line drawn AND the Index of the row where the line is found
def getWordLine0(imLine):
#def getWordLine0(imLine , imageH , imageW):

    #List of summations of non-zero pixels of inverted image for EACH ROW
    rowSum = []
    h , w , l = imLine.shape
    #take Copy of Input Image
    im_row = np.copy(imLine)
    #Invert the Image Copy
    inv = cv2.bitwise_not(im_row) ##
    #Loop on each row to get sum of non-zero pixels of inverted image
    for r in range(0,h):
        row = inv [r:r+1 , 0:w-1 ] ## inv -> im_row
        x = np.count_nonzero (row)
        rowSum.append(x)
    #get the INDEX of Peak Value of all Summations where the line is found
    rowIndex = rowSum.index(max(rowSum))
    # Draw the Line on the Input image
    cv2.line(imLine,(0,rowIndex),(w-1,rowIndex),(0,0,255),2)

    return imLine , rowIndex
