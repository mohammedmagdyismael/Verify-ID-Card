# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:35:53 2018

@author: mohammed-PC
"""

import cv2
import numpy as np


def bk_getSIFTMatch(original , tempelate):
    
    img2 = original
    h,w,_ = img2.shape

    img1 = tempelate
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    h,w = img1.shape
    MIN_MATCH_COUNT = 140
    
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
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
        if m.distance < 0.5*n.distance:
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
        
        pnts = len( np.int32(dst) )
        pnts_lst_x = []
        pnts_lst_y = []
        if pnts == 4:
            for i in range(0, pnts):
                pnts_lst_x.append(np.int32(dst)[i][0][0]) 
                pnts_lst_y.append(np.int32(dst)[i][0][1]) 
        
        min_x = min(pnts_lst_x)
        max_x = max(pnts_lst_x)
        
        min_y = min(pnts_lst_y)
        max_y = max(pnts_lst_y) 
        
        cv2.rectangle(img2, (min_x , min_y), (max_x , max_y), 255, thickness=-1, lineType=8, shift=0)
    
    else:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        
        pnts = len( np.int32(dst) )
        pnts_lst_x = []
        pnts_lst_y = []
        if pnts == 4:
            for i in range(0, pnts):
                pnts_lst_x.append(np.int32(dst)[i][0][0]) 
                pnts_lst_y.append(np.int32(dst)[i][0][1]) 
        
        min_x = min(pnts_lst_x)
        max_x = max(pnts_lst_x)
        
        min_y = min(pnts_lst_y)
        max_y = max(pnts_lst_y) 
        
        cv2.rectangle(img2, (min_x , min_y), (max_x , max_y), 255, thickness=-1, lineType=8, shift=0)
        
        matchesMask = None
    
    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    
    im_matches = img3
    im_crop = cv2.cvtColor(img2 , cv2.COLOR_GRAY2BGR)
    return im_crop , im_matches , min_x , max_x
