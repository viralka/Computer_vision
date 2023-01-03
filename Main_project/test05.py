from re import I, L
import cv2 as cv
from cv2 import BORDER_WRAP
import numpy as np
import image_opener as ip

imgo = cv.imread(r"D:\agnext\Agnext\OpenCv\Main_project\dataset\IMG_20220614_142059.jpg")
img = ip.resizeimg(imgo, 1000)
imggr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imggr = cv.GaussianBlur(imggr, (5, 5), 0)
imggr = cv.erode(imggr, None, iterations=10)
imggr = cv.dilate(imggr, None, iterations=10)

# rest, imgth = cv.threshold(imggr, 0, 255, cv.THRESH_BINARY)
# cv.imshow("window", img)



# thers, img = cv.threshold(imggr, 0, 255, cv.THRESH_BINARY )

# radius = 10
# blank = np.zeros(img.shape[:2], dtype="uint8")
# mask1 = cv.circle(blank.copy(),(1,1), radius, 225, -1)
# area = np.sum(mask1)
# print(area)
# masked = cv.bitwise_and(img, img, mask=mask1)

# mask = np.zeros(img.shape[:2], dtype="uint8")

# roi_corne = np.array([(123,234),(345,456),(289,567)], dtype="int32")

# channel_count = img.shape[2]
# ignore_mask = (255)*channel_count
# cv.fillPoly(mask, [roi_corne], 255)

# masked_image = cv.bitwise_and(img, img, mask=mask)

# cv.imshow("masked_image",masked_image)

count , _ = cv.findContours(imggr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
def rcut(img, roi_corne):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    roi_corne = np.array(roi_corne, dtype="int32")
    channel_count = img.shape[2]
    ignore_mask = (255)*channel_count
    cv.fillPoly(mask, [roi_corne], 255)

    masked_image = cv.bitwise_and(img, img, mask=mask)

    return masked_image


def indexf(count, points):
    for i in range(len(count)):
        for j in range(len(count[i])):
            for k in range(len(count[i][j])):
                if count[i][j][k][0] == points[0] and count[i][j][k][1] == points[1]:
                    return (i,j,k)

def slice_img(count, pi, pf):
    list_of_points = []
    for i in range(pi[0], pf[0]+1):
        for j in range(pi[1], pf[1]+1):
            for k in range(pi[2], pf[2]+1):
                list_of_points.append(list(count[i][j][k]))
    

def rcut(img, roi_corne):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    roi_corne = np.array(roi_corne, dtype="int32")
    channel_count = img.shape[2]
    ignore_mask = (255)*channel_count
    cv.fillPoly(mask, [roi_corne], 255)

    masked_image = cv.bitwise_and(img, img, mask=mask)

    return masked_image

masked_image = rcut(img, )
cv.imshow("masked_image",masked_image)

    
            
# cv.drawContours(img, count, -1, (0, 255, 0), 3)
# cv.imshow("window", img)
# falat_count = np.f
# ind = np.where(count == 351 )
# print(indexf(count, (6,279)))


print(slice_img(count, (0,0,0),(1,1,0) ))



cv.waitKey(0)