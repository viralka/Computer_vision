from ast import Import
from itertools import count
from re import I
from this import d
from xml.dom.expatbuilder import theDOMImplementation
from construct import list_
import cv2 as cv
import numpy as np
from torch import le
import image_opener as ip
from math import atan, dist


def skeleton(img): # optimize it later
    img = img.copy() # for preserving the original image
    imgp1 = cv.GaussianBlur(imggr, (5,5), 0)
    imgp3 = cv.GaussianBlur(imgp1, (5,5), 0)
    imgp4 = cv.erode(imgp3, None, iterations=7)
    imgp5 = cv.dilate(imgp4, None, iterations=7)
    imgp4 = cv.erode(imgp3, None, iterations=9)
    imgp5 = cv.dilate(imgp4, None, iterations=9)
    _, imgp6 = cv.threshold(imgp5, 0, 255, cv.THRESH_BINARY)

    imgp7 = cv.erode(imgp6, None, iterations=6)
    imgp8 = cv.dilate(imgp7, None, iterations=6)

    thinned = cv.ximgproc.thinning(imgp8)
    return thinned

def find_intresting_points(img, n, lent, angle_lim ):
    try:
        imggr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    except:
        imggr = img

    count , _ = cv.findContours(imggr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    count = sorted(count, key=cv.contourArea, reverse=True)
    count = count[:1]

    prob_point = []
    dead = []
    for i in count:
        for j in range(len(i)):
            x, y = i[j].ravel()
            f1= f2 = True

            if (x,y) in dead:
                pass
                
            else:
                try:
                    x_prev, y_prev = i[j-n].ravel()
                    prev_angle= atan((y-y_prev)/(x-x_prev))
                except:
                    f1 = False

                try:
                    x_for, y_for = i[j+n].ravel()
                    forward_angle = atan((y_for-y)/(x_for-x))
                except:
                    f2 = False

                if f1 and f2:
                    if abs(prev_angle-forward_angle) >= angle_lim:
                        prob_point.append((x,y))
                        e = 0
                        while True: # killing nebours
                            try:
                                x_n, y_n = i[j+e].ravel()
                                dist = np.sqrt((x-x_n)**2 + (y-y_n)**2)

                                if dist < lent:
                                    dead.append((x_n,y_n))
                                    e+=1
                                else:
                                    break
                            except:
                                break
                            # cv.circle(img, (x,y), 1, (0,255,0), -1)
    if prob_point[1] in dead:
        pass
    return prob_point





# opening the image 
imgo = cv.imread(r"D:\agnext\Agnext\OpenCv\Main_project\test_data\2.jpg")
img = ip.resizeimg(imgo, 500)
imggr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# making skeleton
thinned = skeleton(imggr)
blank = np.zeros(thinned.shape, np.uint8)
# cv.imshow("skel", thinned)
count , her = cv.findContours(thinned, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(thinned, count, -1, (0,255,0), -1)

cv.imwrite("contour.png", blank)
print(count)


# edge preservetion filter



pon_skel = find_intresting_points(thinned,2, 15, 1)

# mkarking the points on skeleton
for pont in pon_skel:
    x, y = pont
    cv.circle(img, (x,y), 3, (0,0,212), -1)

# print(pon_skel)
pon_img = find_intresting_points(img, lent = 40, angle_lim = .5, n =18)

neares_point = [[point, []] for point in pon_skel]



distlist = []
for intersection_point_dic in neares_point:
    inter_point = intersection_point_dic[0]

    for point in pon_img:
        dist = np.sqrt((inter_point[0]-point[0])**2 + (inter_point[1]-point[1])**2)
        distlist.append((inter_point, point,int( dist)))



sorted_list = sorted(distlist, key=lambda x: x[2])


for inter_point in neares_point:
    for point in sorted_list:
        xi, yi = inter_point[0]
        x, y = point[0]

        if x == xi and y == yi and len(inter_point[1]) < 4:
            inter_point[1].append(point[1])
    


for point in neares_point[7][1]:
    cv.circle(img, point, 3, (200,0,0), -1)




# cv.imshow("contour", img)
# cv.imshow("window", img)
cv.waitKey(0)