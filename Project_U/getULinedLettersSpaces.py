# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:17:50 2018

@author: mohammed-PC
"""


#Optimize the Row of Pixels Plot according to the ColsSum and define the coordinates where Underlined Letters are found
#Input: ColsSum and pixelsRow Lists
#Return: ULinedLettersSpaces List wher Underline Letters are found and new Optimized pixelsRow
def getULinedLettersSpaces (colsSum, pixelsRow):

    ULinedLettersSpaces = []
    for i in range (0 , len(pixelsRow)):
        temp0 = pixelsRow[i]
        if temp0 == 0 :
            temp1 = colsSum [i]
            if temp1 > 0 :
                pixelsRow[i] = 255
                ULinedLettersSpaces.append(i)
            elif temp1 == 0:
                pixelsRow[i] = 0

    return ULinedLettersSpaces, pixelsRow