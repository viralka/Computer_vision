import cv2 as cv
import numpy as np
import image_opener as ip


def image_opener():
    pass
def total_area(img):
    pass
def red_percentage(img, total_area):
    pass


# opening all the images from a folder
img_list = ip.allimg(r'apple_masking\apple_image')
img = img_list[9]
cv.imshow( 'im`',img)

# taking the image 
gray  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# threashoalding for total area
threshold , thres = cv.threshold(gray, 10, 225, cv.THRESH_BINARY)
# cv.imshow("threshold", thres)
# area_T = cv.countNonZero(thres)

#finding conture of it 
contours, hierarchy = cv.findContours(thres,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours, -1, (0, 255, 255), 1)


# # now finding area in the contours

area_T = cv.contourArea(contours[0])
print(area_T,"total area")

# now masking image for less error
img = cv.bitwise_and(img, img , mask= thres)
cv.imshow("img",img)

# converitn all the images in hsv format and bluring a litte
hsv_og = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hsv = cv.GaussianBlur(hsv_og,(5,5),0)
cv.imshow("img",img)

# defining color values

# Red
lower_r = np.array([15, 0, 0])
upper_r = np.array([116, 255, 255])

# createing a mask 
mask = cv.inRange(hsv, lower_r, upper_r)
# cv.imshow("mask", mask)

# finding the conture 
contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours, -1, (0, 255, 0), 1)
cv.imshow("img",img)

# now finding area in the contours
area = cv.contourArea(contours[0])
print(area, "of red")

if area_T != 0 and area_T > area:
    percent= (area/area_T)*100
    print("percent of red area is = ", percent) 

else : 
    print("There is a error")

cv.waitKey(00)