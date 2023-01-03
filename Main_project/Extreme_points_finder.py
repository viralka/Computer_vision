from re import I
import cv2 as cv
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import image_opener as io 


def angle(p1,p2):
    return np.arctan2(p2[1]-p1[1],p2[0]-p1[0])
def f(img):
    cont, her = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    count = max(cont, key = cv.contourArea)

    cv.drawContours(mask, count, -1, (225, 0 , 0), 1)

    top = tuple(count[count[:,:,1].argmin()][0])
    bottom = tuple(count[count[:,:,1].argmax()][0])
    left = tuple(count[count[:,:,0].argmin()][0])
    right = tuple(count[count[:,:,0].argmax()][0])


    return (top, bottom, left, right,angle(top, bottom),angle(left, right))




img = cv.imread(r'Main_project\test_data\obj141_0.jpg')
img1 = io.resizeimg(img, 1000)
img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
mask = np.zeros(img.shape, dtype=np.uint8)


# # img = cv.GaussianBlur(img, (1,1),0)
# img = cv.erode(img, (3,3),iterations=7)
# img = cv.dilate(img, (3,3),iterations=7)

top, bottom, left, right, angletb,angle1tb = extreme_points(img)

cv.line(img1 , top, bottom, (0,0,255), 2)
cv.line(img1 , left, right, (0,0,255), 2)

cv.imshow('img',img1)
img = cv.Canny(img, 100, 200)
cv.imshow("Canny",img)

count , _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

name = "contours2.jpg"
cv.imwrite(name, mask)


cv.waitKey(00)