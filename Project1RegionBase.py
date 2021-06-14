# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 18:04:01 2021

@author: ADMIN
"""

import cv2
import numpy as np

def preprocessRegionBase(path):
    img = cv2.imread(path, 0)
    img = cv2.medianBlur(img,3)
    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3
        


def removeNoise(img, itErode, itDilate):
    kernel = np.ones((3,3),np.uint8)
    img_erode = cv2.erode(img,kernel,iterations = itErode)
    img_dilate = cv2.dilate(img_erode, kernel, iterations = itDilate)
    return img_dilate


def imShowComponents(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('Region Base', labeled_img)
    

def objectCountRegionBase(path, version, itErode, itDilate):
    th3 = preprocessRegionBase(path)
    if version == 1:
        th3 = cv2.bitwise_not(th3)
    th3_dilate = removeNoise(th3, itErode, itDilate)
    num_labels, labels_im = cv2.connectedComponents(th3_dilate)
    print(num_labels)
    imShowComponents(labels_im)
    cv2.imshow("Original", cv2.imread(path))

    
    cv2.waitKey(0)
    
path = ["1_wIXlvBeAFtNVgJd49VObgQ_sinus.png", "1_wIXlvBeAFtNVgJd49VObgQ.png", "1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png", "1_zd6ypc20QAIFMzrbCmJRMg.png","objets1.jpg", "objets2.jpg", "objets3.jpg", "objets4.jpg"]   

index = 2
itErode = [1,1,1,1,2,2,2,2]
itDilate = [1,1,1,1,10,10,10,10]
version = [2,0,0,0,1,1,1,1]

objectCountRegionBase(path[index], version[index], itErode[index], itDilate[index])