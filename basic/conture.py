import cv2 as cv
import numpy as np


image = cv.imread("OpenCv\Images\deer.jpg")
img = cv.resize(image,(450,672))
cv.imshow("OG", img)

blank = np.zeros(img.shape, dtype='uint8')


gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
# gray = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

# # Finding conture with countour methord 
# cv.imshow("gray", gray)

canny = cv.Canny(gray, 125, 175)
cv.imshow("canny", canny)
 
# contour are python list of all the contours in the image
# hierarchy outher most is is high in haracy the inner is low in haracy 
# RETR_LIST return all the contours in the image
# RETR_EXTERNAL return all the outer contours in the image
# RETR_TREE return all the inner contours in the image
# SEE documentation 
contour, hierarchy = cv.findContours(canny,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# draw contour
cv.drawContours(blank, contour, -1, (225, 0 , 0), 1)
cv.imshow("blank", blank)
# print(len(contour))


# draw contour by threshold 
ret , thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow("Threshold", thresh)


cv.waitKey(0)