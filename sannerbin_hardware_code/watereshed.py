from tkinter import image_names
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import dev_Tools as dt

image_path = r'C:\Users\mridu\Downloads\nullframe2.jpg'
img = cv.imread(image_path)
# dt.hsv_finder(image_path)
gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

ret, thresh = cv.threshold(gray,0,25,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# dt.bg_remover_segmentation(image_path, lowH = 0, highH = 255, lowS = 75, highS = 255, lowV = 0, highV = 255)


# converitn all the images in hsv format and bluring a litte
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# hsv = cv.GaussianBlur(hsv_og,(11,11),0)
cv.imshow("img",img)
# defining color values
# mask value
lower = np.array([7, 73, 0])
upper = np.array([130, 255, 255])
# createing a mask 
mask = cv.inRange(hsv, lower, upper)
# mask = cv.bitwise_not(mask)

thresh = mask 
# img = cv.bitwise_and(img , img, mask= mask)
# img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# lot_blur = cv.GaussianBlur(img1, (11,11),0)
# # cv.imshow("mask", lot_blur)




cv.imshow('thresh1', thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)


# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)


# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)

img[markers == -1] = [0,0,255]

cv.imshow('thresh', img)

cv.waitKey(0)

