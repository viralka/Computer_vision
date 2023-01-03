from email.mime import image
import cv2 as cv 
import _dev_Tools as dt 
import numpy as np


# dt.hsv_finder(r'D:\agnext\Agnext\OpenCv\_countour_detection\test\2.jpeg')

image = cv.imread(r'D:\agnext\Agnext\OpenCv\_countour_detection\test\2.jpeg')
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

lower_hsv1 = np.array([0, 0, 100])
higher_hsv1 = np.array([255, 255, 125])
mask1 = cv.inRange(hsv, lower_hsv1, higher_hsv1)
mask1 = cv.bitwise_not(mask1)

cv.imwrite("mask2.png", mask1)

mask1 = dt.resizeimg(mask1, 800)

cv.imshow('mask1', mask1)

cv.waitKey(0)