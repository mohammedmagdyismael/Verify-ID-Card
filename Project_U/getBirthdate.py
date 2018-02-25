# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:55:26 2018

@author: mohammed-PC
"""

import cv2
import numpy as np
import datetime 
from matplotlib import pyplot as plt
from keras.models import load_model


#Input : RGB Image(Element of the Returned values of the getMainROIs method)
#Return: String formatted Number
def getBirthdate(image_slice):
    im = image_slice
    copy = im
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(im, (9, 9), 0)
    edges = cv2.Canny(im,140,200)
    kernel = np.ones((27,27), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges, contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))

    lst = []
    slices = []
    #newview = cv2.resize(copy, (900, 400))

    for i in range(0,len(contours)):
        arr = [x,y,w,h] = cv2.boundingRect(contours[i])
        x,y,w,h = cv2.boundingRect(contours[i])
        lst.append(arr)
        y0 = lst[i][1]
        x0 = lst[i][0]
        yf0 = (lst[i][1] + lst[i][3] )
        xf0 = (lst[i][0] + lst[i][2] )
        slices.append(copy [y0:yf0 , x0:xf0 ])

    '''
    if lst[0][1] > lst[1][1]:
        cv2.imshow("image",slices[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imshow("image",slices[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

    im = slices[1]
    clr = im
    im =cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist,bins = np.histogram(im.flatten(),256,[0,256])
    hist.cumsum()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im)
    im = clahe.apply(im)
    im = cv2.bitwise_not(im)

    low = np.array([150])
    upper = np.array([255])
    im = cv2.inRange(im ,low , upper )
    im, contours, hierarchy = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    lstt = []
    sli = []

    for i in range(0,len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 10.0:
            arr = [x,y,w,h] = cv2.boundingRect(contours[i])
            lstt.append(arr)
            #print (area)
    lstt.sort()
    space = 0
    id_no = []
    #print(lstt)

    classifier = load_model('ar_numbers_v2.h5')
    classifier.compile(loss= 'categorical_crossentropy', optimizer='adam')

    for i in range(0,len(lstt)) :

        y0 = lstt[i][1] - space
        x0 = lstt[i][0] - space
        yf0 = (lstt[i][1] + lstt[i][3] + space)
        xf0 = (lstt[i][0] + lstt[i][2] + space)
        slice_img = clr [y0:yf0 , x0:xf0 ]
        slice_img = cv2.resize(slice_img, (30, 30))
        slice_img = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
        (thresh, slice_img) = cv2.threshold(slice_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        now =  datetime.datetime.now()
        y = str(now.date().year)
        m = str(now.date().month)
        d = str(now.date().day)
        h = str(now.hour)
        mi = str(now.minute)
        s = str(now.second)
        name = "Dvx_"+y+m+d+"_"+h+mi+s
     #   cv2.imwrite(name+"_"+str(i).zfill(2)+'.jpg',slice_img)
        slice_img = cv2.cvtColor(slice_img,cv2.COLOR_GRAY2RGB)
        imresize = cv2.resize(slice_img, (32 , 32))
        imlist = np.array([imresize])
        classes =classifier.predict_classes(imlist)
        id_no.append(classes[0])


    plt.imshow(clr)
    final_birth = str(id_no[0])+str(id_no[1])+str(id_no[2]) \
            +str(id_no[3])+"-"+str(id_no[5]) \
            +str(id_no[6])+"-"+str(id_no[8]) \
            +str(id_no[9])
    final_birth_lst = [id_no[2] , id_no[3] , id_no[5] , id_no[6] , id_no[8] , id_no[9]]
    return final_birth , final_birth_lst
