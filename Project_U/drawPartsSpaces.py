# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:23:51 2018

@author: mohammed-PC
"""

from matplotlib import pyplot as plt


def drawPartsSpaces (pixelsRowOPT , image):
    cnt = 0
    for i in range (0 , len(pixelsRowOPT)):
        if pixelsRowOPT[i] == 0 and cnt == 0 :
            image [0:h , i:i+1] = (0,255,255)
            cnt = 1
        elif pixelsRowOPT[i] != 0  :
            cnt = 0

    plt.figure()
    plt.imshow(image)