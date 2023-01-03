import cv2 as cv
import numpy as np

image = cv.imread('OpenCv\Images\orange_bird.jpg')
img = cv.resize(image, (500,333), interpolation=cv.INTER_CUBIC )
cv.imshow("bird", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', gray)

# Lapalacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap) 

# Saddel 
sabelx = cv.Sobel(gray, cv.CV_64F, 1, 0)   # x direction
sabelly = cv.Sobel(gray, cv.CV_64F, 0, 1)   # y direction
combined = cv.bitwise_and(sabelx, sabelly)  # combine x and y direction

cv.imshow('X direction', sabelx)
cv.imshow('Y direction', sabelly)
cv.imshow('Combined', combined)

# Canny
canny = cv.Canny(gray, 100, 200)
cv.imshow('Canny', canny)



cv.waitKey(00)