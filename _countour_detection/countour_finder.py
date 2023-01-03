from email.mime import image
import cv2 as cv 
import numpy as np
import _dev_Tools as dt
import time 



image = cv.imread(r'D:\agnext\Agnext\OpenCv\_countour_detection\14.png')

hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

lower_hsv1 = np.array([32, 0, 0])
higher_hsv1 = np.array([255, 255, 255])
mask1 = cv.inRange(hsv, lower_hsv1, higher_hsv1)
mask1 = cv.bitwise_not(mask1)



t1 = time.time()

image = mask1

p1 = cv.erode(image, (3,3), iterations= 10)
p2 = cv.dilate(p1, (3,3), iterations= 10)

img = image


count , her = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# cv.drawContours(img, count, -1, (0,255,0), 2)

print(len(count))


t2 = time.time()
print("Time used : ",t2 - t1)

cv.imshow("image", img)
cv.waitKey(0)