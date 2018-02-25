# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:38:30 2018

@author: mohammed-PC
"""

import cv2
import numpy as np

def bk_getStatments(nai):
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
