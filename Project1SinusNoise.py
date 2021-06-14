# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:05:43 2021

@author: ADMIN
"""
import cv2
import numpy as np

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    obj_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area>100:
            obj_count+=1
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
    return obj_count

scale = 1
delta = 0
ddepth = cv2.CV_16S
window_name = ('Sobel Demo - Simple Edge Detector')

path= "1_wIXlvBeAFtNVgJd49VObgQ_sinus.png"

img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
src = cv2.GaussianBlur(img, (3, 3), 0)

imgBlur = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

grad_x = cv2.Sobel(imgBlur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
grad_y = cv2.Sobel(imgBlur, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

#

img = cv2.medianBlur(grad,3)

#global threashould
#https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
ret1,th1 = cv2.threshold(grad,127,255,cv2.THRESH_BINARY)
ret2,th2 = cv2.threshold(grad,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(grad,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("1", th3)
kernel = np.ones((2,2),np.uint8)
th3_erode = cv2.erode(th3,kernel,iterations = 1)
    
th3_dilate = cv2.dilate(th3_erode, kernel, iterations = 6)

num_labels, labels_im = cv2.connectedComponents(th3_dilate)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img
    
    
print(num_labels)
cv2.imshow("source", cv2.imread(path))
labeled_img = imshow_components(labels_im)
cv2.imshow("labeled", labeled_img)

cv2.waitKey(0)

