import cv2 
import numpy as np


#đọc file
#path= "1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png"
path = "objets2.jpg"
img = cv2.imread(path,0)

#tăng đồ tương phan
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

#thêm bộ lọc 
img = cv2.medianBlur(img,3)

#global threashould
#https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

'''
#adapptive threashould
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
'''

#xử lý hậu phân vùng




cv2.imshow("Original",  cv2.imread(path))

th3 = cv2.bitwise_not(th3)
kernel = np.ones((2,2),np.uint8)
th3_erode = cv2.erode(th3,kernel,iterations = 1)
th3_dilate = cv2.dilate(th3_erode, kernel, iterations = 7)
#cv2.imshow("Move noise 1", th3)

'''
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(th3,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow("tách ", sure_fg)
'''
num_labels, labels_im = cv2.connectedComponents(th3_dilate)
'''
num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(th3,10)
labels_im = labels_im -1
sizes = stats[1:, -1]

#loại bỏ vùng nhỏ
min_size = 150
img2 = np.zeros((labels_im.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, num_labels-1):
    if sizes[i] >= min_size:
        img2[labels_im == i + 1] = 255

cv2.imshow("Remove small object",img2)
'''
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()
    
print(num_labels-1)
imshow_components(labels_im)

cv2.waitKey(0)