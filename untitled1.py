#https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
import cv2 as cv
img = cv.imread('1_wIXlvBeAFtNVgJd49VObgQ.png',0)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv.imshow("test", cl1)
cv.imshow("test1", img)
cv.waitKey(0)