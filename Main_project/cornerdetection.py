from re import I
import cv2 as cv
from cv2 import BORDER_WRAP
import numpy as np
import image_opener as ip

imgo = cv.imread(r"D:\agnext\Agnext\0010100.png")
# img = ip.resizeimg(imgo, 2000)
img = imgo
imggr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imggr = cv.GaussianBlur(imggr, (5, 5), 0)
imggr = cv.erode(imggr, None, iterations=10)
imggr = cv.dilate(imggr, None, iterations=10)

rest, img = cv.threshold(imggr, 0, 255, cv.THRESH_BINARY)
cv.imshow("window", img)


# # Haris corner detection # good for geometric edges
# imggr = np.float32(imggr)
# harris = cv.cornerHarris(imggr, 2, 3, 0.04)
# harris = cv.dilate(harris, None)
# img[ harris> 0.01 * harris.max()] = [0, 0, 255]
# cv.imshow("harris",img)


# # goodFeaturesToTrack
corner =cv.goodFeaturesToTrack(imggr, 100,0.01,20) # img, how many corner, quality, min distance between corners
#print(corner)
corner = np.int0(corner)
imgo1 = imgo.copy()
for i in corner:
    x,y = i.ravel()
    cv.circle(imgo1,(x,y),3,255,-1)

cv.imshow("goodFeaturesToTrack",imgo1)


# imgo2 = imgo.copy()
# corner =cv.cornerEigenValsAndVecs(imggr, 11, 3) # img, how many corner, quality, min distance between corners
# print(corner.shape)
# corner = np.int0(corner)

# for i in corner:
#     x,y= i.ravel()
#     cv.circle(imgo2,(x,y),3,255,-1)

# cv.imshow("cornerEigenValsAndVecs",imgo2)

# goodFeaturesToTrack
# corner =cv.cornerSubPix(imggr, 100,0.01,(-1,-1)) # img, how many corner, quality, min distance between corners
# print(corner)
# # corner = np.int0(corner)
# imgo3 = imgo.copy()

# for i in corner:
#     x,y = i.ravel()
#     cv.circle(imgo3,(x,y),3,255,-1)

# cv.imshow("cornerSubPix",imgo3)


# # Initiate FAST object with default values
# fast = cv.FastFeatureDetector_create()
# # find and draw the keypoints
# kp = fast.detect(img,None)
# img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# # Print all default params
# # print( "Threshold: {}".format(fast.getThreshold()) )
# # print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
# # print( "neighborhood: {}".format(fast.getType()) )
# # print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
# # cv.imwrite('fast_true.png', img2)
# # Disable nonmaxSuppression
# fast.setNonmaxSuppression(0)
# kp = fast.detect(img, None)
# # print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
# img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

# cv.imshow('fast_false.png', img3)


# # rice detachment 

thers, imggr = cv.threshold(imggr, 0, 255, cv.THRESH_BINARY )

count, her = cv.findContours(imggr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   
# print(count)
# print(her)
cv.drawContours(img, count, -1, (0,255,0), 1)

for i in count:
    for j in i:
        x,y = j.ravel()
        cv.circle(imgo,(x,y),10,(0,0,255),1)
# cv.imshow("window",imgo)
cv.waitKey(0)



print(count[0][1])

print(cv.contourArea(count[0]))


cv.waitKey(0)