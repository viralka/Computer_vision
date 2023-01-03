import cv2 as cv
import numpy as np

image = cv.imread('OpenCv\Images\girl.jpg')
img = cv.resize(image, (500,700), interpolation=cv.INTER_CUBIC)
cv.imshow('OG',img)

# split the image into color chanels
b,g,r = cv.split(img)

cv.imshow('b',b)
cv.imshow('g',g)
cv.imshow('r',r)

# makeing a blank image
z = np.zeros(img.shape[:2], dtype = "uint8")

# # merge the image
# merge = cv.merge([b,g,r])
# cv.imshow('merge',merge)

# individual color chanels 

blue = cv.merge([b,z,z]) 
cv.imshow('blue', blue)

green = cv.merge([z,g,z])
cv.imshow('green', green)

red = cv.merge([z,z,r])
cv.imshow('red', red)


print(img.shape)
print()

cv.waitKey(2200)