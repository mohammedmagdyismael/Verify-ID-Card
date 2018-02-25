# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:27:25 2018

@author: mohammed-PC
"""

import cv2
import datetime 
import numpy as np
from matplotlib import pyplot as plt

import getVerticals as gv
import getWordLine0 as gw0
import getPixelsRow as gpr
import getWordLineSpaces as gwls
import getULinedLettersSpaces as guls

def segmentLetters(im):

    im = cv2.resize(im,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    imh , imw , _ = im.shape
    imG = cv2.cvtColor(im ,cv2.COLOR_BGR2GRAY )
    ret2, imG = cv2.threshold(imG ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imGinv = cv2.bitwise_not(imG)
    ret,thresh1 = cv2.threshold(imGinv,127,255,cv2.THRESH_BINARY)
    kernel = np.ones((2,2),np.uint8)
    im = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

    op_imLine , baseline = gw0.getWordLine0(im.copy() , imh , imw)
    print("Word Line is at Row Index: ",baseline)
    op_pixelsRow = gpr.getPixelsRow(im.copy(),baseline)
    op_PixelsRowSpaces = gwls.getWordLineSpaces(op_pixelsRow)
    op_colsSum, op_colsSpaces , op_maxPeak = gv.getVerticals(im.copy() , (1,1))
    op_ULinedLettersSpaces, op_pixelsRowOPT = guls.getULinedLettersSpaces(op_colsSum,op_pixelsRow)

    _, contours,_ = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    bigC = []
    smallC = []
    XofRect = []

    for i in range (0 , len (contours)  ):
        x,y,w,h = cv2.boundingRect(contours[i])
        yRange = y + h
        if baseline in range (y , yRange):
            bigC.append(i)
            XofRect.append(x)
        else :
            smallC.append(i)

    XofRect.sort()

    img = np.zeros([imh,imw,3],dtype=np.uint8)
    img.fill(255)
    for i in range ( 0 , len (bigC)):
        cv2.drawContours(img, [contours[bigC[i]]], -1,(0),cv2.FILLED)
    upperBorderPoints = [0] * imw
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img , (int(imw/2),int(imh)))
    img = cv2.resize(img , (int(imw),int(imh)))

    for c in range(0, imw):
        col = img[0:imh, c:c + 1]
        for i in range(0, len(col)):
            x = col[i][0]
            if x != 255:
                upperBorderPoints[c] = baseline - i
                break

    # status 0 ....stable
    # status 1 .... curves up
    # status -1 ..... curves down
    curveStatus = [0] * imw
    for i in range(1, len(upperBorderPoints) - 1):
        bkwd = upperBorderPoints[i] - upperBorderPoints[i - 1]

        if bkwd == 0:
            curveStatus[i - 1] = 0
        if bkwd > 0:
            curveStatus[i - 1] = 1
        if bkwd < 0:
            curveStatus[i - 1] = -1

    hand = []
    flg = 0
    for i in range(len(curveStatus) - 1, -1, -1):
        s = curveStatus[i]
        if s == 1 and flg == 0:
            flg = 1
        if s == -1 and flg == 1:
            hand.append([i + 1, upperBorderPoints[i + 1]])
            flg = 0

    a = np.array(hand)
    x = a[:, 0] #x pos of local minimum point to cut at
    y = a[:, 1] #upper border hight


    plt.figure()
    plt.plot(upperBorderPoints)
    plt.plot(x, y, 'ro')
#
#    plt.show()
#    plt.savefig('books_read.png')
    imgcpy = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    newcuts = []

    imGG = cv2.cvtColor(imG , cv2.COLOR_GRAY2BGR)
    for i in range(0, len(x)):
        if x[i] in op_ULinedLettersSpaces:
            newcuts.append(x[i])
            imgcpy[0:imh, int(x[i]) -2: int(x[i]) - 1] = (0, 0, 255)
            imGG [0:imh, int(x[i]) -2 : int(x[i]) -1 ] = (0, 0, 255)
    for i in range(0, len(XofRect)):
            imgcpy[0:imh, int(XofRect[i]) -2 : int(XofRect[i]) -1] = (0, 0, 255)
            imGG [0:imh, int(XofRect[i]) -2: int(XofRect[i]) -1] = (0, 0, 255)

    now = datetime.datetime.now()
    y = str(now.date().year)
    m = str(now.date().month)
    d = str(now.date().day)
    hr = str(now.hour)
    mi = str(now.minute)
    s = str(now.second)
    name = "Dvx_" + y + m + d + "_" + hr + mi + s
    cv2.imwrite(name+".jpg" , imgcpy)

    now = datetime.datetime.now()
    y = str(now.date().year)
    m = str(now.date().month)
    d = str(now.date().day)
    hr = str(now.hour)
    mi = str(now.minute)
    s = str(now.second)
    name = "Dvx_" + y + m + d + "_" + hr + mi + s
    cv2.imwrite(name+".png" , imGG)
    cv2.imshow("cuts", imgcpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return newcuts