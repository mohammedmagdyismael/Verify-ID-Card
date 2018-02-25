#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 08:10:11 2018

@author: mgd-pc
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_temp(front , back , scale):
    
    #Scale
    scale_factor = scale
    
    #Real Card width and height
    w = 8.6
    h = 5.5
    
    #init. card area parameters
    scaled_h = int(h*scale_factor)
    scaled_w = int(w*scale_factor)
    
    ##Front
    #X
    photo_xi , photo_xf = int(0.3*scale_factor) , int((0.3+2.2)*scale_factor)
    birthdate_xi , birthdate_xf =  int(0.3*scale_factor) , int((0.3+3)*scale_factor)
    pin_xi , pin_xf =  int(0.3*scale_factor) , int((0.3+3)*scale_factor)
    logo_xi , logo_xf = int((0.3+3)*scale_factor) , int(7.5*scale_factor)
    name_xi , name_xf = int((0.3+3)*scale_factor) , int(w*scale_factor)
    add_xi , add_xf = int((0.3+3)*scale_factor) , int(w*scale_factor)
    idno_xi , idno_xf = int((0.3+3)*scale_factor) , int(w*scale_factor)
    #Y
    photo_yi , photo_yf = int(0.4*scale_factor) , int(3.2*scale_factor)
    birthdate_yi , birthdate_yf =  int(3.7*scale_factor) , int(4.8*scale_factor)
    pin_yi , pin_yf =  int(4.8*scale_factor) , int(h*scale_factor)
    logo_yi , logo_yf = int(0*scale_factor) , int(1.2*scale_factor)
    name_yi , name_yf = int(1.3*scale_factor) , int(2.5*scale_factor)
    add_yi , add_yf = int(2.5*scale_factor) , int(3.8*scale_factor)
    idno_yi , idno_yf = int(4.1*scale_factor) , int(4.9*scale_factor)
    
    ##Back
    #X
    nesr_xi , nesr_xf =  int(7*scale_factor) , int(8.3*scale_factor)
    phar_xi , phar_xf = int(0.3*scale_factor) , int(2*scale_factor)
    rest_xi , rest_xf = int(2.4*scale_factor) , int(7*scale_factor)
    expiry_xi , expiry_xf = int(2.5*scale_factor) , int(7*scale_factor)
    code_xi , code_xf = int(0*scale_factor) , int(w*scale_factor)
    #Y
    nesr_yi , nesr_yf =  int(0*scale_factor) , int(2.2*scale_factor)
    phar_yi , phar_yf = int(0*scale_factor) , int(2.2*scale_factor)
    rest_yi , rest_yf = int(0.3*scale_factor) , int(2.2*scale_factor)
    expiry_yi , expiry_yf = int(2.4*scale_factor) , int(3*scale_factor)
    code_yi , code_yf = int(3*scale_factor) , int(h*scale_factor)
    
    
    card = cv2.bitwise_not(np.zeros((scaled_h,scaled_w,3), np.uint8))
    
    card_f = front
    card_f = cv2.resize(card_f, (scaled_w , scaled_h)) 
    
    card_b = back
    card_b = cv2.resize(card_b, (scaled_w , scaled_h)) 
    
    thick = 2
    #front 
    photo = card_f[photo_yi:photo_yf , photo_xi:photo_xf]
    birthdate = card_f[birthdate_yi:birthdate_yf , birthdate_xi:birthdate_xf]
    pin = card_f[pin_yi:pin_yf , pin_xi:pin_xf]
    logo = card_f[logo_yi:logo_yf , logo_xi:logo_xf]
    name = card_f[name_yi:name_yf , name_xi:name_xf]
    add = card_f[add_yi:add_yf , add_xi:add_xf]
    idno = card_f[idno_yi:idno_yf , idno_xi:idno_xf]
    
    front = [photo, birthdate, pin, logo, name, add, idno]
#    for i in front:
#        cv2.imshow("Image" , i)
#        cv2.waitKey(0)

    cv2.rectangle(card_f,(photo_xi,photo_yi),(photo_xf,photo_yf),(0,0,255) , thick)
    cv2.rectangle(card_f,(birthdate_xi,birthdate_yi),(birthdate_xf,birthdate_yf),(0,0,0) , thick)    
    cv2.rectangle(card_f,(pin_xi,pin_yi),(pin_xf,pin_yf),(0,255,0) , thick)    
    cv2.rectangle(card_f,(logo_xi,logo_yi),(logo_xf,logo_yf),(255,0,0) , thick)    
    cv2.rectangle(card_f,(name_xi,name_yi),(name_xf,name_yf),(50,0,50) , thick)    
    cv2.rectangle(card_f,(add_xi,add_yi),(add_xf,add_yf),(0,50,50) , thick)    
    cv2.rectangle(card_f,(idno_xi,idno_yi),(idno_xf,idno_yf),(50,50,0) , thick)
    
    #back
    nesr = card_b[nesr_yi:nesr_yf , nesr_xi:nesr_xf]    
    phar = card_b[phar_yi:phar_yf , phar_xi:phar_xf ]    
    rest = card_b[rest_yi:rest_yf , rest_xi:rest_xf]
    code = card_b[code_yi:code_yf , code_xi:code_xf]
    expiry = card_b[expiry_yi:expiry_yf , expiry_xi:expiry_xf] 
    
    back = [nesr, phar, rest, code, expiry]
#    for i in back:
#        cv2.imshow("Image" , i)
#        cv2.waitKey(0)


    cv2.rectangle(card_b,(nesr_xi,nesr_yi),(nesr_xf,nesr_yf),(0,0,255) , thick)
    cv2.rectangle(card_b,(phar_xi,phar_yi),(phar_xf,phar_yf),(0,0,0) , thick)
    cv2.rectangle(card_b,(rest_xi,rest_yi),(rest_xf,rest_yf),(0,255,0) , thick)
    cv2.rectangle(card_b,(code_xi,code_yi),(code_xf,code_yf),(255,0,0) , thick)
    cv2.rectangle(card_b,(expiry_xi,expiry_yi),(expiry_xf,expiry_yf),(50,0,50) , thick)
    
    
    cv2.destroyAllWindows()
    
    return front, back , card_f, card_b















