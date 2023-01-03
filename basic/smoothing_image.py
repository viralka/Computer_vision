import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread('OpenCv\Images\orange_bird.jpg')
img = cv.resize(image, (500,333), interpolation=cv.INTER_CUBIC )
cv.imshow("OG", img)

# Average Blur
blur = cv.blur(img, (5,5))
blur = cv.Canny(blur, 125,175)
cv.imshow('Average Blur', blur)

# Gaussian Blur
blur = cv.GaussianBlur(img, (5,5), 0)
blur = cv.Canny(blur, 125,175)
cv.imshow('Gaussian Blur', blur)

# Median Blurr
blur = cv.medianBlur(img, 5)
blur = cv.Canny(blur, 125,175)
cv.imshow('Median Blur', blur)

# Bilateral Blur (image, diameter, sigmaColor (how much color will effect the pixels), sigmaSpace (how much space will effect the pixels)))) 
blur = cv.bilateralFilter(img, 9, 75, 75)
blur = cv.Canny(blur, 125,175)
cv.imshow('Bilateral Blur', blur)


cv.waitKey(00)