#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:50:40 2018

@author: mgd-pc
"""

###########################Libraries##########################################
import cv2 
import numpy as np 
from keras.models import load_model
from matplotlib import pyplot as plt
######################User-Defined Modules####################################
import getFace as gf
import isIDCARD as idc
import dataSplit as dsp
import getIDNumber as gid
import getMainROIs as groi
import getBirthdate as gbd
import bk_getNumber as bkgn
import bk_getSIFTMatch as bkgs
import bk_getStatments as bkgst
import bk_getMainROIs as bkgroi
import longestSubstring as lsub
from Id_Crop_template import *
from setPrespectives import *
###############################################################################
###############################Main Method#####################################
###############################################################################
if __name__ == "__main__":
#def main (front_path , back_path ):
    #Change to Front&Back Files 
    file_front = "id00.jpg"
    file_back = "id1.jpg"
    
    #Variables
    files = [file_front , file_back]
    output = []
    points_presp = []
    drawing=False
    mode=True
    ############################################################
    #####################Mouse Events###########################
    ############################################################    
    def interactive_drawing(event,x,y,flags,param):
        global ix,iy,drawing, mode
    
        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True
            ix,iy=x,y
            if len(points_presp) < 4 :
                cv2.line(img,(x,y), (x,y) , (0,0,255) , 8 )   
                points_presp.append([x,y])
        
        elif event==cv2.EVENT_LBUTTONUP:
            drawing=False 
        return x,y
    ############################################################
    ##################Crop to Prespectives######################
    ############################################################
        
    for file in files:
        img = cv2.imread(file) 
        h , w , _ = img.shape
        img = cv2.resize(img , (int(w/4),int(h/4)))
        
        cv2.namedWindow('Photo')
        cv2.setMouseCallback('Photo',interactive_drawing)
        
        while(1):
            cv2.imshow('Photo',img)
            k=cv2.waitKey(1)&0xFF
            if k==27:    #esc --> exit
                break
            elif k==110: #n --> new
                img = cv2.imread(file)
                h , w , _ = img.shape
                img = cv2.resize(img , (int(w/4),int(h/4)))
                points_presp = []
            elif k==99: #c --> crop
                 if len(points_presp) == 4:
                     img = cv2.imread(file) 
                     h , w , _ = img.shape
                     img = cv2.resize(img , (int(w/4),int(h/4)))
                     im = setPrespectives(img,points_presp)
                     img = im
                     output.append(img)
                     points_presp = [] 
                             
        cv2.destroyAllWindows()
        if files.index(file) == 0 and len(output) == 0:
            break
    
#############################################################################
#############################################################################    
    print("===================Results=====================")
    if len(output) == 2:
        print ("Full Card Modified To New Prespectives")
        fr_arr, bck_arr, fr, bck = crop_temp(output[0] , output[1] , 80)
        
        face = gf.getFace(fr_arr[0]) 
        
        check_card = idc.isIDCARD(fr_arr[3]) 
        
        dateS , dateL = gbd.getBirthdate(fr_arr[1]) #
        
        idnoS, idnoL = gid.getIDNumber(fr_arr[6]) #
        
        print("===================Front=====================")
        print("IS CARD?: " ,check_card)
        print("BirthDate:" , dateS)
        print("ID NUMBER: " ,idnoS)
    
        if dateL == idnoL :
            print("Check Matching BirthDate and ID Number :" + "TRUE")
        else:
            print("Check Matching BirthDate and ID Number :" + "FALSE")
            

        temp_nesr = cv2.imread("logo_1.jpg")
        h , w , _ = temp_nesr.shape
        temp_nesr = cv2.resize(temp_nesr, (int(w/2) ,int(h/2)) )
        im_crop, new_im_matches ,end_x , _  = bkgs.bk_getSIFTMatch(bck_arr[0] , temp_nesr)
        
        temp_pharaoh = cv2.imread("logo_2.jpg")
        h , w , _ = temp_pharaoh.shape
        temp_pharaoh = cv2.resize(temp_pharaoh, (int(w/2) ,int(h/2)) )
        new_im_crop, new_im_matches,_,start_x= bkgs.bk_getSIFTMatch(bck_arr[1] , temp_pharaoh)
        
        out = bkgst.bk_getStatments(bck_arr[2])
        
        h , w , _ = out[0].shape
        im_id = cv2.resize(out[0], (int(w*2) ,int(h*2)) ) 
        
        h , w , _ = bck_arr[4].shape
        im_exp = cv2.resize(bck_arr[4], (int(w*2) ,int(h*2)) )  
        
        out_slice = bkgroi.bk_getMainROIs(im_id)
        txt , id_list = bkgn.bk_getNumber(out_slice[0]) #
        avg = (len(txt) + len (idnoS))/2
        perc = len(lsub.longestSubstring(txt , idnoS)) / avg
      
        out_slice = bkgroi.bk_getMainROIs(im_exp) 
        _ , exp_list = bkgn.bk_getNumber(out_slice[0]) #
        final_no = str(exp_list[0])+str(exp_list[1])+str(exp_list[2]) \
                +str(exp_list[3])+" / "+str(exp_list[5])+str(exp_list[6])+" / " \
                +str(exp_list[8])+str(exp_list[9])
        
        print("===================Back======================")
        check = False
        if perc > 0.50 :
            print("Front/Back Validation ?: " , True)   
            check = True
        else:
            check = False
        print("Expiry Date: " + final_no )
        
        
        output = []
        output.append(idnoS)    #ID number
        output.append(check)    #Validate Front/Back
        output.append(final_no) #Expiry Date
     #   return output
        

    else:
        print ("Missing Card Modifications !!")
 
