#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 06:26:51 2017

@author: mgd
"""

import cv2
import numpy as np
import datetime
import glob
import collections
from matplotlib import pyplot as plt
from keras.models import load_model

##############################################################################
#Input : RGB Image
#Return: RGB Image
def setPrespectives(image_file):
    def order_points(pts):
    	rect = np.zeros((4, 2), dtype = "float32")
    	s = pts.sum(axis = 1)
    	rect[0] = pts[np.argmin(s)]
    	rect[2] = pts[np.argmax(s)]
    	diff = np.diff(pts, axis = 1)
    	rect[1] = pts[np.argmin(diff)]
    	rect[3] = pts[np.argmax(diff)]
    	return rect

    def four_point_transform(image, pts):
    	rect = order_points(pts)
    	(tl, tr, br, bl) = rect
    	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    	maxWidth = max(int(widthA), int(widthB))
    	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    	maxHeight = max(int(heightA), int(heightB))
    	dst = np.array([
    		[0, 0],
    		[maxWidth - 1, 0],
    		[maxWidth - 1, maxHeight - 1],
    		[0, maxHeight - 1]], dtype = "float32")
    	M = cv2.getPerspectiveTransform(rect, dst)
    	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    	return warped

    img = cv2.imread(image_file)
    clr = img
    img = cv2.resize(img,(1000,500))
    clr = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 60, 10)
    image, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    for c in cnts:
    	peri = cv2.arcLength(c, True)
    	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    	if len(approx) == 4:
    		screenCnt = approx
    		break
    ratio = clr.shape[0] / 500.0
    warped = four_point_transform(clr, screenCnt.reshape(4, 2) * ratio)

    h , w , l = warped.shape
    h = int( w/1.5 )
    warped = cv2.resize(warped , (w , h))

    return warped

##############################################################################
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
##############################################################################
#Input : RGB Image
#Return: List RGB Images
def getMainROIs(image_file):
    im = image_file
    copy = im
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(im, (9, 9), 0)
    edges = cv2.Canny(im,140,200)
    edges = cv2.resize(edges, (900, 400))
    kernel = np.ones((67,67), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges, contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    lst = []
    slices = []
    newview = cv2.resize(copy, (900, 400))

    for i in range(0,len(contours)):
        arr = [x,y,w,h] = cv2.boundingRect(contours[i])
        x,y,w,h = cv2.boundingRect(contours[i])
        lst.append(arr)
        y0 = lst[i][1] - 3
        if y0 < 0:
            y0 = lst[i][1]
        x0 = lst[i][0]
        yf0 = (y0 + lst[i][3] )
        xf0 = (x0 + lst[i][2] )
        slices.append(newview [y0:yf0, x0:xf0 ])

    return slices

##############################################################################
#Input : RGB Image(Element of the Returned values of the getMainROIs method)
#Return: BOOL Values
def isIDCARD(image_slices):
    try:
        MIN_MATCH_COUNT = 140
        img1 = cv2.imread('logo_0.jpg',0) # Main reference
        h , w = img1.shape
        img2 = image_slices
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = cv2.resize(img2 , (w, h))

        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            print ("Is ID CARD found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            return True

        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None
            return False

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

        plt.imshow(img3, 'gray'),plt.show()
    except ValueError:
        return False

##############################################################################
#Input : RGB Image(Element of the Returned values of the getMainROIs method)
#Return: List of RGB Images
def dataSplit(image_slice):
    newview1 = image_slice
    copy1 = image_slice
    newview1 = cv2.fastNlMeansDenoisingColored(newview1,None,10,10,7,21)
    newview1 = cv2.cvtColor(newview1, cv2.COLOR_BGR2GRAY)
    hist,bins = np.histogram(newview1.flatten(),256,[0,256])
    hist.cumsum()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    newview1 = clahe.apply(newview1)
    low = np.array([0])
    upper = np.array([55])
    newview1 = cv2.inRange(newview1 ,low , upper )
    kernel = np.ones((33,33), np.uint8)
    newview1 = cv2.dilate(newview1, kernel, iterations=1)
    newview1, contours1, hierarchy = cv2.findContours(newview1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    lst1 = []
    slices1 = []

    for i in range(0,len(contours1)):
        arr = [x,y,w,h] = cv2.boundingRect(contours1[i])
        x,y,w,h = cv2.boundingRect(contours1[i])
        lst1.append(arr)
        y0 = lst1[i][1]
        x0 = lst1[i][0]
        yf0 = (lst1[i][1] + lst1[i][3] )
        xf0 = (lst1[i][0] + lst1[i][2] )
        slices1.append(copy1 [y0:yf0 , x0:xf0 ])
    return slices1

##############################################################################
#Input : RGB Image(Element of the Returned values of the dataSplit method)
#Return: String formatted Number
def getIDNumber(image_slice):
    classifier = load_model('ar_numbers_v2.h5')
    classifier.compile(loss= 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
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
    upper = np.array([70])
    cl1 = cv2.inRange(cl1 ,low , upper )
    '''
    cv2.imshow("image",cl1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    image, contours, hierarchy = cv2.findContours(cl1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print (len(contours))

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
        now =  datetime.datetime.now()
        y = str(now.date().year)
        m = str(now.date().month)
        d = str(now.date().day)
        h = str(now.hour)
        mi = str(now.minute)
        s = str(now.second)
        name = "Dvx_"+y+m+d+"_"+h+mi+s
      #  cv2.imwrite(name+"_"+str(i).zfill(2)+'.jpg',slice_img)
        slice_img = cv2.cvtColor(slice_img,cv2.COLOR_GRAY2RGB)
        imresize = cv2.resize(slice_img, (32 , 32))
        imlist = np.array([imresize])
        classes =classifier.predict_classes(imlist)
        id_no.append(classes[0])


    plt.imshow(view)
    #print("ID:")
    final_no = str(id_no[0])+str(id_no[1])+str(id_no[2]) \
            +str(id_no[3])+str(id_no[4])+str(id_no[5]) \
            +str(id_no[6])+str(id_no[7])+str(id_no[8]) \
            +str(id_no[9])+str(id_no[10])+str(id_no[11]) \
            +str(id_no[12])+str(id_no[13])
    '''print(str(id_no[0])+str(id_no[1])+str(id_no[2])
            +str(id_no[3])+str(id_no[4])+str(id_no[5])
            +str(id_no[6])+str(id_no[7])+str(id_no[8])
            +str(id_no[9])+str(id_no[10])+str(id_no[11])
            +str(id_no[12])+str(id_no[13]))'''
    final_no_lst = id_no[1:7]
    return final_no , final_no_lst

##############################################################################
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
    classifier.compile(loss= 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

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

def getStatments(nai):
    im = nai.copy()
    h , w, l = im.shape
    h = h *1
    w = w *1
    im_resized = cv2.resize(im, (int(w), int(h)))
    im = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(im,140,200)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    rowSum = []
    inv = edges.copy()
    for r in range(0,h):
        row = inv [r:r+1 , 0:w-1 ]
        x = np.count_nonzero (row)
        rowSum.append(x)

    cuts_pos = []
    cnt = 0
    for i in range (0 , len(rowSum)):
        if rowSum[i] == 0 and cnt == 0:
            #im_verticalSpaces [0:h , i:i+1] = (255,0,0)
            cuts_pos.append(i)
            cnt = 1

        elif rowSum[i] != 0  :
            cnt = 0

    if cuts_pos[0] != 0:
        cuts_pos.insert(0, 0)

    statments = []
    for i in range (0 , len(cuts_pos)-1):
        statments.append(im_resized [cuts_pos[i]:cuts_pos[i+1] , 0:w])
#    plt.figure()
#    plt.plot(rowSum)
#    cv2.imshow("image",statments[0])
#    cv2.imshow("image1",statments[1])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return statments

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
##############################################
#Letters v2###################################
##############################################
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

#Get Word Line Using Hough Line Detector
#Input: Processed Image, Word in Black and White background
#Return: Image of the word with the line drawn AND the Index of the row where the line is found
def getWordLine1(image):
    h , w , l = image.shape
    #convert Input image RGB -> Gray
    gry = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #Get Edges using Canny Edge Detector
    edges = cv2.Canny(gry,50,150,apertureSize = 3)
    #Get Word Line Using Hough Line Detector
    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        #x0 = a*rho
        y0 = b*rho
        #x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        #x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    #Select Pixels Row
    rowIndex = int((y1+y2)/2.0)
    row_of_pixels = image [rowIndex:rowIndex , 0:w-1 ]
    #Invert the Row
    row_of_pixels = cv2.bitwise_not(row_of_pixels)
    #Convert the row to gray scale so it can be thresholded to two values 0/255
    row_of_pixels = cv2.cvtColor(row_of_pixels , cv2.COLOR_BGR2GRAY)
    #loop on each pixel in the row to define its value even 0 or 255
    for i in range(0,len(row_of_pixels[0])):
        temp = row_of_pixels [0][i]
        if temp < 128:
            row_of_pixels [0][i] = 0
        elif temp > 128:
            row_of_pixels [0][i] = 255
    pixelsRow = row_of_pixels[0]

    return pixelsRow  , rowIndex


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

#Get the summations of non zero elements in each column (contains word) and zero elements in each column (spaces)
#Input: Processed Image, Word in Black and White background
#Return: sum of non zeros elements in each column (colsSum) and zero columns which are spaces (colsSpaces)
def getVerticals(image, kernelsize):
    colsSum = []
    colsSpaces = []
    h , w , l = image.shape
    #dilate the image
    kernel = np.ones(kernelsize, np.uint8)
    dil = cv2.dilate(image, kernel, iterations=1)
    #invert it
    inv = cv2.bitwise_not(dil)
    inv = cv2.cvtColor(inv ,  cv2.COLOR_BGR2GRAY) ##
    low = np.array([128])
    upper = np.array([255])
    inv = cv2.inRange(inv ,low , upper )
    plt.figure()
    plt.imshow(inv)
    #get sum of columns from dilated-Inverted image
    for c in range(0,w):
        col = inv [0:h , c:c+1 ]
        x = np.count_nonzero (col)
        colsSum.append(x)
        if x == 0 :
            colsSpaces.append(c)
    maxPeak = max(colsSum)
    return colsSum , colsSpaces ,maxPeak

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


def drawPartsSpaces (pixelsRowOPT , image):
    cnt = 0
    for i in range (0 , len(pixelsRowOPT)):
        if op_pixelsRowOPT[i] == 0 and cnt == 0 :
            image [0:h , i:i+1] = (0,255,255)
            cnt = 1
        elif op_pixelsRowOPT[i] != 0  :
            cnt = 0

    plt.figure()
    plt.imshow(image)

#fit the Image size to the height and width of the statment included
#Input: Image
#Return: new dimensions to be used in resizing
def resize (originaImage):
    colsSum = []
    colsSpaces = []
    h , w , l = originaImage.shape
    #invert it
    inv = cv2.bitwise_not(originaImage)
    inv = cv2.cvtColor(inv ,  cv2.COLOR_BGR2GRAY)
    low = np.array([128])
    upper = np.array([255])
    inv = cv2.inRange(inv ,low , upper )
    #get sum of columns from dilated-Inverted image
    for c in range(0,w):
        col = inv [0:h , c:c+1 ]
        x = np.count_nonzero (col)
        colsSum.append(x)
    rowsSum = []
    for r in range(0,h):
        row = inv [r:r+1 , 0:w ]
        x = np.count_nonzero (row)
        rowsSum.append(x)

    flagL = 0
    for i in range(0,len(colsSum)):
         if colsSum[i] != 0 and flagL == 0:
            alignL = i
            flagL = 1

    flagR = 0
    for i in range(len(colsSum)-1 , -1 , -1):
         if colsSum[i] != 0 and flagR == 0:
            alignR = i
            flagR = 1

    flagU = 0
    for i in range(0,len(rowsSum)):
         if rowsSum[i] != 0 and flagU == 0:
            alignU = i
            flagU = 1

    flagD = 0
    for i in range(len(rowsSum)-1 , -1 , -1):
         if rowsSum[i] != 0 and flagD == 0:
            alignD = i
            flagD = 1
    #
    return alignR , alignL , alignD , alignU


def segmentLetters(im):

    im = cv2.resize(im,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    imh , imw , _ = im.shape
    imG = cv2.cvtColor(im ,cv2.COLOR_BGR2GRAY )
    ret2, imG = cv2.threshold(imG ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imGinv = cv2.bitwise_not(imG)
    ret,thresh1 = cv2.threshold(imGinv,127,255,cv2.THRESH_BINARY)
    kernel = np.ones((2,2),np.uint8)
    im = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

    op_imLine , baseline = getWordLine0(im.copy() , imh , imw)
    print("Word Line is at Row Index: ",baseline)
    op_pixelsRow = getPixelsRow(im.copy(),baseline)
    op_PixelsRowSpaces = getWordLineSpaces(op_pixelsRow)
    op_colsSum, op_colsSpaces , op_maxPeak = getVerticals(im.copy() , (1,1))
    op_ULinedLettersSpaces, op_pixelsRowOPT = getULinedLettersSpaces(op_colsSum,op_pixelsRow)

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

##############################################
#TESTING######################################
##############################################

#Main Script

if __name__ == "__main__":
    raw_image = cv2.imread("id0.jpg")
    h , w , l = raw_image.shape
    raw_image = cv2.resize(raw_image , (int(w/3),int(h/3)))

    #modified_image = setPrespectives("id0.jpg")

    face = getFace(raw_image)

    id_card = cv2.imread("id0.jpg")
    main_areas = getMainROIs(id_card)  ##*

#    for n in range(0,len(main_areas)):
#        cv2.imshow("Part"+str(n),main_areas[n])
    check_card = isIDCARD(main_areas[3])

    dateS , dateL = getBirthdate(main_areas[1])
    NAI = dataSplit(main_areas[2])
    idnoS , idnoL = getIDNumber(NAI[0])
    print("IS CARD?: " ,check_card)
    print("BirthDate:" , dateS)
    print("ID NUMBER: " ,idnoS)

    if dateL == idnoL :
        print("Check Matching BirthDate and ID Number :" + "TRUE")
    else:
        print("Check Matching BirthDate and ID Number :" + "FALSE")
#    cv2.imshow("Raw Image",raw_image)
##    cv2.imshow("Modified Image",modified_image)
#    cv2.imshow("Face",face)
#    cv2.imshow("Name",NAI[2])
#    cv2.imshow("ID Number",NAI[0])
#    cv2.imshow("Address",NAI[1])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    op_AddStatments = getStatments(NAI[1])
    op_addStatment0 = segmentLetters(op_AddStatments[0])
    op_addStatment1 = segmentLetters(op_AddStatments[1])

    op_NAMEStatments = getStatments(NAI[2])
    op_nameStatment0 = segmentLetters(op_NAMEStatments[0])
    op_nameStatment1 = segmentLetters(op_NAMEStatments[1])



##############################################################
    op_AddStatments = getStatments(NAI[1])
    op_NAMEStatments = getStatments(NAI[2])

    op_addStatment0 = bwStatment(op_AddStatments[0])
    op_addStatment1 = bwStatment(op_AddStatments[1])

    op_nameStatment0 = bwStatment(op_NAMEStatments[0])
    op_nameStatment1 = bwStatment(op_NAMEStatments[1])

##############################################################
#    cv2.imwrite("image0.png", op_nameStatment0)
#    cv2.imwrite("image1.png", op_nameStatment1)
#    cv2.imwrite("image2.png", op_addStatment0)
#    cv2.imwrite("image3.png", op_addStatment1)
#
    cv2.imshow("image0", op_nameStatment0)
    cv2.imshow("image1", op_nameStatment1)
    cv2.imshow("image2", op_addStatment0)
    cv2.imshow("image3", op_addStatment1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
##############################################
### OCR NAME AND ADDRESS IN PROGRESS##########
##############################################

    im = op_nameStatment1.copy()

    im=cv2.imread("jj.png")
#    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#    im = cv2.fastNlMeansDenoisingColored(im,None,10,10,7,21)
#    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#    blurred = cv2.GaussianBlur(im, (55, 55), 2)
#    '''low = np.array([0])
#    upper = np.array([70])
#    blurred = cv2.inRange(blurred ,low , upper )
#    plt.imshow(blurred)'''
#    im = cv2.adaptiveThreshold(blurred,100,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,39,5)
#    hist,bins = np.histogram(im.flatten(),256,[0,256])
#    cdf = hist.cumsum()
#    cdf_normalized = cdf * hist.max()/ cdf.max()
#    plt.hist(im.flatten(),256,[0,256], color = 'r')
#    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#    cl1 = clahe.apply(im)
#    #cl1 = cv2.bitwise_not(cl1)
#    low = np.array([0])
#    upper = np.array([70])
#    im = cv2.inRange(cl1 ,low , upper )
#    im =cv2.cvtColor(im , cv2.COLOR_GRAY2BGR)
#    im = cv2.bitwise_not(im)



#    plt.figure()
#    plt.imshow(im)
#    op_alighR, op_alighL, op_alighD, op_alighU = resize(im)
#    print(op_alighR, op_alighL, op_alighD, op_alighU)
#    im = im [8:290 , 12:132]
#    plt.figure()
#    plt.imshow(im)
    h , w , lyr = im.shape

    #copies of the image
    #im_wordLine1 = np.copy(im)
    im_wordLine0 = np.copy(im)
    im_wordVerticalSpaces = np.copy(im)
    im_verticalSpaces = np.copy(im)

    #get and visualized image and index of row where the line is selected
    op_imLine , op_rowIndex = getWordLine0(im_wordLine0)
#    plt.figure()
#    plt.imshow(op_imLine)
    print("Word Line is at Row Index: ",op_rowIndex)

    #get row of pixels of Word Line
    op_pixelsRow = getPixelsRow(im,op_rowIndex)
    plt.figure()
    plt.plot(op_pixelsRow)

    #get array of spaces in the Word Line
    op_PixelsRowSpaces = getWordLineSpaces(op_pixelsRow)
    #plt.figure()
    #plt.plot(op_PixelsRowSpaces)

    #get Vertical spaces between discontinued ranges
    #op_VerticalSum before Dilate
    op_colsSum, op_colsSpaces , op_maxPeak = getVerticals(im_verticalSpaces , (1,1))
    print(op_maxPeak)
    print(op_colsSum)
    plt.figure()
    plt.plot(op_colsSum)
    #plt.figure()
    #plt.plot(op_colsSpaces)


    #Optimize the Horizontal spaces (periods of Underlined Letters) using the
    #Vertical Spaces
    op_ULinedLettersSpaces, op_pixelsRowOPT = getULinedLettersSpaces(op_colsSum,op_pixelsRow)
    #print(op_ULinedLettersSpaces)

    #draw spaces between parts of the word
    #drawPartsSpaces(op_pixelsRowOPT, im_verticalSpaces)

    ##
    repeated =[item for item, count in collections.Counter(op_colsSum).items() if count > 1]
    repeatedRecord = []
    for i in range (0,len(repeated)):
        #print("Value:", repeated[i]  ,"  Repeated:",op_colsSum.count(repeated[i]))
        repeatedRecord.append([repeated[i], op_colsSum.count(repeated[i])])
    repeatedRecord = sorted(repeatedRecord,key=lambda x: (x[1]),reverse= True)


    colsSumOptimizer = repeatedRecord [0][0]
    if repeatedRecord [0][0] == 0 :
        colsSumOptimizer = repeatedRecord [1][0]

    print ("Optimizer: " ,colsSumOptimizer)

    plt.figure()
    plt.plot(op_colsSum)

    op_colsSumOPT = []
    for i in range (0 , len(op_colsSum)):
        op_colsSumOPT.append( op_colsSum [i] - (colsSumOptimizer))
        if op_colsSumOPT [i] < 0 :
            op_colsSumOPT [i] = 0
    plt.figure()
    plt.plot(op_colsSumOPT)
    #CUTS >>>>>>>>
    cnt = 0
    cuts_pos = []
    for i in range (0 , len(op_colsSumOPT)):
        if op_colsSumOPT[i] == 0 and  i in op_ULinedLettersSpaces :
            continue

        if op_colsSumOPT[i] == 0 and i in op_colsSpaces and cnt == 0:
            #im_verticalSpaces [0:h , i:i+1] = (255,0,0)
            cuts_pos.append(i)
            cnt = 1

        if op_colsSumOPT[i] == 0 and i not in op_colsSpaces and i not in op_ULinedLettersSpaces and cnt == 0:
            #im_verticalSpaces [0:h , i:i+1] = (255,0,0)
            cuts_pos.append(i)
            cnt = 1
        elif op_colsSumOPT[i] != 0  :
            cnt = 0

    cuts_pos.append(w)

    contNslices = []
    for i in range (0,len(cuts_pos) - 1):
        temp_slice = im_verticalSpaces [0:h , cuts_pos[i]:cuts_pos[i+1]]
        cpy = np.copy(temp_slice)
        hcpy , wcpy , _ = cpy.shape
        #shoukd be enhanced
        im = cv2.fastNlMeansDenoisingColored(temp_slice,None,10,10,7,21)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(im, (55, 55), 2)
        im = cv2.adaptiveThreshold(im,100,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,39,5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(im)
        #cl1 = cv2.bitwise_not(cl1)
        low = np.array([0])
        upper = np.array([70])
        cl1 = cv2.inRange(cl1 ,low , upper )
        imcont, contours,_ = cv2.findContours(cl1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        contNslices.append([len(contours) , 0 ])



    print (cuts_pos)
    print (op_colsSum)
    print (op_maxPeak)
    for i in range (0,len(cuts_pos) - 1):
        print( max ( op_colsSum[ cuts_pos[i] : cuts_pos[i + 1]  ] ) )
        contNslices [i] [1] = max ( op_colsSum[ cuts_pos[i] : cuts_pos[i + 1]  ] )


    for i in range (0 , len (contNslices)):
        if contNslices [i][0] > 1 and (contNslices [i][1] / op_maxPeak) > 0.5 :
            print("")

        if contNslices [i][0] == 1 and (contNslices [i][1] / op_maxPeak) < 0.5 :
            if i <= len(cuts_pos) - 2:
                cuts_pos.remove(cuts_pos[i])



    for i in range (0 , len(cuts_pos)):
        im_verticalSpaces [0:h , cuts_pos[i]: cuts_pos[i] + 1] = (255,0,0)

    print (contNslices)
    plt.figure()
    plt.imshow(im_verticalSpaces)

#    cv2.imwrite("new3.png",im_verticalSpaces)
