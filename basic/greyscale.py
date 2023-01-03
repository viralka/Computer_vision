import cv2 as cv 

img = cv.imread('Images\clock tower.jpg')

# cv.imshow('ct', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

blur = cv.GaussianBlur(img, (11,11), cv.BORDER_DEFAULT)
cv.imshow('blur', blur)

## Edge Cascade Like super cool stuff
canny = cv.Canny(img,125, 175)
gcanny = cv.Canny(gray,125,175)

cv.imshow('canny',canny)
cv.imshow('canny',canny)

## Dilating the image
dill = cv.dilate(canny, (3,3), iterations=1)
cv.imshow('Dilated', dill)

## Erode the image
erode = cv.erode(dill, (3,3), iterations=1)
cv.imshow('erode',erode)

## Resize the image
resize = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC )
cv.imshow('resize',resize)

## crop the image
crop = img[100:300, 100:400]
cv.imshow('crop', crop)

cv.waitKey(0)