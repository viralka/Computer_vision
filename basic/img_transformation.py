import cv2 as cv
import numpy as np


image = cv.imread('OpenCv\Images\girl.jpg')

img = cv.resize(image, (500,700), interpolation=cv.INTER_CUBIC )
cv.imshow('resize',img)


def translate(ima, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (ima.shape[1], ima.shape[0])
    return cv.warpAffine(ima, transMat, dimensions)


def rotate( img, angle, rotCenter = None):
    hight , width = img.shape[:2]

    if rotCenter is None:
        rotCenter = (hight//2, width//2)
    
    rotMat = cv.getRotationMatrix2D(rotCenter ,angle,1.0)

    return cv.warpAffine(img,rotMat,rotCenter)

# # Resizeing
# resize = cv.resize(img, (300,300))

# flipping
# 1 = horizontaly 
# 0 = vitericaly 
#-1 = both
flip = cv.flip(img, 0)
cv.imshow("img", flip)

# crop image
crop = img[200:400, 300:400]

trans = rotate(img,-1)
cv.imshow('translate', trans) 

cv.waitKey(0)