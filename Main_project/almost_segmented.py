from re import I
import cv2 as cv
from cv2 import BORDER_WRAP
import numpy as np
import image_opener as ip
import cutting_img_by_corner as cutter
from math import atan

def skeleton(img, n=10): # optimize it later
    img = img.copy() # for preserving the original image
    imgp1 = cv.GaussianBlur(imggr, (5,5), 0)
    imgp3 = cv.GaussianBlur(imgp1, (5,5), 0)
    imgp4 = cv.erode(imgp3, None, iterations=n)
    imgp5 = cv.dilate(imgp4, None, iterations=n)
    imgp4 = cv.erode(imgp3, None, iterations=n)
    imgp5 = cv.dilate(imgp4, None, iterations=n)
    _, imgp6 = cv.threshold(imgp5, 0, 255, cv.THRESH_BINARY)

    imgp7 = cv.erode(imgp6, None, iterations=n)
    imgp8 = cv.dilate(imgp7, None, iterations=n)

    thinned = cv.ximgproc.thinning(imgp8)

    return thinned


imgo = cv.imread(r"D:\agnext\Agnext\OpenCv\Main_project\test_data\7.jpg")
imguo = ip.resizeimg(imgo, 500)
imggr = cv.cvtColor(imguo, cv.COLOR_BGR2GRAY)
imggr = cv.GaussianBlur(imggr, (5, 5), 0)
imggr = cv.erode(imggr, None, iterations=10)
imggr = cv.dilate(imggr, None, iterations=10)

rest, imgth = cv.threshold(imggr, 0, 255, cv.THRESH_BINARY)
# cv.imshow("window", img)
thers, img = cv.threshold(imggr, 0, 255, cv.THRESH_BINARY )

count, her = cv.findContours(imggr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   
# cv.drawContours(imguo, count, -1, (25,0,0), 1)
radius = 7
corners = []
for i in count:
    for j in i:
        x,y = j.ravel()
        blank = np.zeros(img.shape[:2], dtype="uint8")
        mask1 = cv.circle(blank.copy(),(x,y), radius, 225, -1)
        areaog = np.sum(mask1)
        masked = cv.bitwise_and(img, img, mask=mask1)
        area = np.sum(masked)
        ratio_area = area/areaog
        try :
            dist = np.sqrt((x-corners[-1][0])**2 + (y-corners[-1][1])**2)
            if dist < radius and ratio_area > 0.5:
                continue

        except:
            pass

        if ratio_area >= 0.75 or ratio_area <= 0.25:
            corners.append((x,y))

# print(corners)

for corner in corners:
    cv.circle(imguo, corner, radius, (0,255,0), -1)




thinned = skeleton(img)
# cv.imshow("skel", thinned)


count , her = cv.findContours(thinned, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(imguo, count, -1, (20,0,220), 1)

# cv.imshow("contour", img)
inters  = cv.goodFeaturesToTrack(thinned,4,0.01,20)
inters = np.int64(inters)

for inter in inters:
    x, y = inter.ravel()
    cv.circle(imguo, (x,y), 3, (200,0,0), -1)




neares_point = []
for point in inters:
    x, y = point.ravel()
    neares_point.append([(x,y),[]])

duo_list = []
for corner in corners:
    for inter in inters:
        x, y = inter.ravel()
        dist = np.sqrt((x-corner[0])**2 + (y-corner[1])**2)
        duo_list.append((list(inter), list(corner),dist))


duo_list_s = sorted(duo_list,key=lambda l:l[2], reverse=False)

cv.imshow("window", imguo)
# print(duo_list_s)

cutting_points= [[i,[]] for i in inters]
for  i in duo_list_s:
    for point in cutting_points:
        x, y = point[0].ravel()
    
        print(i[0][0])
        if i[0][0][0] == x and i[0][0][1] == y and len(point[1]) <4:
            point[1].append(i[1])
            
        else:
            continue

# print(cutting_points)
            
i=0
for point in cutting_points:
    if len(point[1]) == 4:
        print(point[1])
        masked=cutter.rcut(imguo, point[1])
        
        cv.imshow(str(i), masked)
        i+=1


# print(cutting_points)
# for i in  duo_list_s:
#     for j in cutting_points:
        
#             # print(i[1][1])
#             # t=list(j[0])
#             print(j[0][0][0][1])
#             break
#         # if i[1][0]== j[0] and i[1][1]== j[1]:
#             print(i[1],j)
            
       

#     x1, y1 = i[1]
#     cutting_points.append((x1,y1))

# print(cutting_points)

# cuted = cutter.rcut(imguo, cutting_points)

# cv.imshow("window1", cuted)

cv.waitKey(0)
cv.destroyAllWindows()
