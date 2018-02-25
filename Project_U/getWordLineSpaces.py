# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:14:13 2018

@author: mohammed-PC
"""

#Get the zero elements (Spaces) in the Pixels Row (Word Line)
#Input: PixelsRow (Word Line)
#Return: List X coordinates of the Spaces in the PixelsRow
def getWordLineSpaces(WordLine_arr):
    PixelsRowSpaces = []
    #Loop on each pixel in the PixelsRow, Store the position of the zero element (spaces)
    for i in range(0, len(WordLine_arr)):
        if WordLine_arr[i] == 0 :
            PixelsRowSpaces.append(i)
    return PixelsRowSpaces