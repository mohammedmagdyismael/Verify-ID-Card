# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:45:49 2018

@author: mohammed-PC
"""
import cv2
import numpy as np

#Input : RGB Image
#Return: RGB Image
def getFace(image_file):
    im = image_file
    im_gry = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hist,bins = np.histogram(im_gry.flatten(),256,[0,256])
    hist.cumsum()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im_gry = clahe.apply(im_gry)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(im_gry, 1.3, 5)
    for (x,y,w,h) in faces:
        #im = cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = im[y:y+h, x:x+w]
        #roi_color = cv2.resize(roi_color, (200,200))

    return roi_color