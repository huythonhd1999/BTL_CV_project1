# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 18:04:01 2021

@author: ADMIN
"""

import cv2
import numpy as np

def preprocessContourBase(path):
    img = cv2.imread(path)
    imgContour = img.copy()
 
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #imgBlur = cv2.GaussianBlur(imgGray,(3,3),1)

    imgBlur = cv2.medianBlur(imgGray,3)#dùng khi ảnh có nhiễu dạng muối tiêu
    #ret2,th2 = cv2.threshold(imgBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    imgCanny = cv2.Canny(imgBlur,50,50)
    

    return imgCanny, imgContour

def removeNoise(img, itErode, itDilate):
    kernel = np.ones((2,2),np.uint8)
    img_erode = cv2.erode(img,kernel,iterations = itErode)
    img_dilate = cv2.dilate(img_erode, kernel, iterations = itDilate)
    return img_dilate


def getObjectContoursBase(img, minSize, imgContour):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
   
    obj_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>minSize:
            obj_count+=1
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
    return  imgContour, obj_count

def objectCountContourBase(path, minSize):
    imgCanny, imgContour = preprocessContourBase(path)
    #imgCannyRemoveNoise = removeNoise(imgCanny, itErode, itDilate)
    imgContour, obj_count = getObjectContoursBase(imgCanny, minSize, imgContour)
    
    
    print(obj_count)
    cv2.imshow("Original", cv2.imread(path))
    cv2.imshow("Canny", imgCanny)
    cv2.imshow("Contour base", imgContour)
    
    cv2.waitKey(0)
    
path = ["1_wIXlvBeAFtNVgJd49VObgQ_sinus.png", "1_wIXlvBeAFtNVgJd49VObgQ.png", "1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png", "1_zd6ypc20QAIFMzrbCmJRMg.png","objets1.jpg", "objets2.jpg", "objets3.jpg", "objets4.jpg"]   
minSize = [15, 12, 10, 2, 40, 80, 200, 80] 


index = 2

objectCountContourBase(path[index], minSize[index])