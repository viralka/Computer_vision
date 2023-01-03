import cv2 as cv

image = cv.imread("OpenCv\Images\deer.jpg")
img = cv.resize(image,(450,672))

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)

# simple thresholding 
threshold , thres = cv.threshold(gray, 90, 225, cv.THRESH_BINARY)
cv.imshow("threshold", thres)

# inverse thresholding
threshold , thres = cv.threshold(gray, 90, 225, cv.THRESH_BINARY_INV)
cv.imshow("threshold_inverse", thres)


# adaptive thresholding
adaptive_threshold = cv.adaptiveThreshold(gray,225,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,3)
cv.imshow("adaptive_threshold",adaptive_threshold) 

cv.waitKey(2200)
