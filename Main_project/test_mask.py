from re import I
import cv2 as cv
from cv2 import BORDER_WRAP
import numpy as np
import image_opener as ip

imgo = cv.imread(r"D:\agnext\Agnext\OpenCv\apple_obj0.jpg")
img = ip.resizeimg(imgo, 500)
imggr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imggr = cv.GaussianBlur(imggr, (5, 5), 0)
imggr = cv.erode(imggr, None, iterations=10)
imggr = cv.dilate(imggr, None, iterations=10)

rest, imgth = cv.threshold(imggr, 0, 255, cv.THRESH_BINARY)
cv.imshow("window", img)



thers, imggr = cv.threshold(imggr, 0, 255, cv.THRESH_BINARY )

cv.imshow("window", imggr)

count, her = cv.findContours(imggr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   
# print(count)
# print(her)
cv.drawContours(img, count, -1, (0,255,0), 1)

#! 


blank = np.zeros(img.shape[:2], dtype="uint8")

mask = cv.circle(blank.copy(),(200,270), 100, 225, -1)

mask1 = cv.rectangle(blank.copy(), (50,50),(450,450), 225, -1)
cv.imshow("m", mask)

# superimpose the mask on image
masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow("masked", masked)

# rectangle mask on image 
masked = cv.bitwise_and(img, img, mask=mask1)
cv.imshow("masked1", masked)

cv.waitKey(2200)