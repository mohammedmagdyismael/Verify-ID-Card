# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:11:46 2018

@author: mohammed-PC
"""

import cv2
import numpy as np

def bwStatment(statment):
    im = statment.copy()
    h , w , l = im.shape
    raw_image = cv2.resize(im.copy() , (int(w*3),int(h*3)))
    imRaw_gry = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    imRaw_gry = cv2.blur(imRaw_gry, (5, 5))
    edged = cv2.Canny(imRaw_gry, 50 , 90)
    kernel = np.ones((5,5),np.uint8)
    gradient = cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, kernel)
    image, contours, hierarchy=cv2.findContours(gradient.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros([h*3,w*3,3],dtype=np.uint8)
    img.fill(255)
    img = cv2.drawContours(img, contours, -1,(0),-1)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)

    return dilation