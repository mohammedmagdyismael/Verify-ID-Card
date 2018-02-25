# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:47:05 2018

@author: mohammed-PC
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from keras.models import load_model

#Input : RGB Image(Element of the Returned values of the dataSplit method)
#Return: String formatted Number
def bk_getNumber(image_slice):
    classifier = load_model('ar_numbers_v2.h5')
    classifier.compile(loss= 'categorical_crossentropy', optimizer='adam')
    im = image_slice
    im = cv2.resize(im , (1742, 241))
    view =im
    color = im
    im = cv2.fastNlMeansDenoisingColored(im,None,10,10,7,21)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(im, (55, 55), 2)
    '''low = np.array([0])
    upper = np.array([70])
    blurred = cv2.inRange(blurred ,low , upper )
    plt.imshow(blurred)'''
    im = cv2.adaptiveThreshold(blurred,100,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,39,5)
    hist,bins = np.histogram(im.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.hist(im.flatten(),256,[0,256], color = 'r')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(im)
    #cl1 = cv2.bitwise_not(cl1)
    low = np.array([0])
    upper = np.array([100]) 
    cl1 = cv2.inRange(cl1 ,low , upper )
    '''
    cv2.imshow("image",cl1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    image, contours, hierarchy = cv2.findContours(cl1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#    print (len(contours))

    space = 0
    lst = []

    for i in range(0,len(contours)):
        newview = view
        arr = [x,y,w,h] = cv2.boundingRect(contours[i])
        lst.append(arr)

    lst.sort()
    id_no = []
    for i in range(0,len(contours)) :
        y0 = lst[i][1] - space
        x0 = lst[i][0] - space
        yf0 = (lst[i][1] + lst[i][3] + space)
        xf0 = (lst[i][0] + lst[i][2] + space)
        slice_img = newview [y0:yf0 , x0:xf0 ]
        slice_img = cv2.resize(slice_img, (30, 30))
        slice_img = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
        (thresh, slice_img) = cv2.threshold(slice_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        slice_img = cv2.cvtColor(slice_img,cv2.COLOR_GRAY2RGB)
        imresize = cv2.resize(slice_img, (32 , 32))
        imlist = np.array([imresize])
        classes =classifier.predict_classes(imlist)
        id_no.append(classes[0])


#    plt.imshow(view)
    #print("ID:")
#    final_no = str(id_no[0])+str(id_no[1])+str(id_no[2]) \
#            +str(id_no[3])+str(id_no[4])+str(id_no[5]) \
#            +str(id_no[6])+str(id_no[7])+str(id_no[8]) \
#            +str(id_no[9])+str(id_no[10])+str(id_no[11]) \
#            +str(id_no[12])+str(id_no[13])
#    '''print(str(id_no[0])+str(id_no[1])+str(id_no[2])
#            +str(id_no[3])+str(id_no[4])+str(id_no[5])
#            +str(id_no[6])+str(id_no[7])+str(id_no[8])
#            +str(id_no[9])+str(id_no[10])+str(id_no[11])
#            +str(id_no[12])+str(id_no[13]))'''
    txt = ''
    for i in range (0,len(id_no)):
         txt += str(id_no[i])
#    print(txt)
    return txt , id_no