import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Different image opend

image = cv.imread("OpenCv\Images\deer.jpg")
img = cv.resize(image,(450,672))
# cv.imshow("deer",img)

image = cv.imread('OpenCv\Images\girl.jpg')
img2 = cv.resize(image, (500,700), interpolation=cv.INTER_CUBIC)
# cv.imshow('Girl',img2)

image = cv.imread('OpenCv\Images\orange_bird.jpg')
img3 = cv.resize(image, (500,333), interpolation=cv.INTER_CUBIC )
cv.imshow("bird", img3)

img_list = [img, img2, img3]
for img in img_list:

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('img', gray)
    gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Histogram for grey")
    plt.xlabel("Number of pixels")
    plt.ylabel("Brightness of pixels")
    plt.plot(gray_hist)
    plt.xlim([0, 256])
    plt.show()
    

# making a blank image
blank = np.zeros(img.shape[:2], dtype = "uint8")
mask = cv.rectangle(blank.copy(), (100,150),(400,500), 225, -1)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# masking image 
masked = cv.bitwise_and(img, img , mask= mask)

# mkaing its histogram 
cv.imshow("masked", masked)
gray_hist = cv.calcHist([masked], [0], None, [256], [0, 256])

# draw histogram
plt.figure()
plt.title("Histogram for grey")
plt.xlabel("Number of pixels")
plt.ylabel("Brightness of pixels")
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

# colored histogram

for img in img_list:

    plt.figure()
    plt.title("Histogram for colored image")
    plt.xlabel("Number of pixels")
    plt.ylabel("Brightness of pixels")


    color = ('b','r','g')
    for i,col in enumerate(color):
        cv.imshow("image with color", img)
        hist = cv.calcHist([img],[i], None, [256], [0,256])
        plt.plot(hist, color = col)
        plt.xlim([0, 256])

    plt.show()

    cv.waitKey(0) 