import cv2 as cv
import numpy as np

image = cv.imread('OpenCv\Images\girl.jpg')

img = cv.resize(image, (500,700), interpolation=cv.INTER_CUBIC )
cv.imshow('resize',img)

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