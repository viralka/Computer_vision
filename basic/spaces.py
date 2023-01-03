import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread('OpenCv\Images\orange_bird.jpg')
img = cv.resize(image, (500,333), interpolation=cv.INTER_CUBIC )

cv.imshow("OG", img)

# BGR to RGB conversion
rbg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rbg)

# BGR to Grayscale conversion
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', gray)

# BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# BGR to HSV conversion
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# Reading image with Matplotlib
plt.imshow(img)
plt.show()

# Now correct image via RGB conversion
plt.imshow(rbg)
plt.show()

cv.waitKey(2200) 