import os 
import subprocess
import cv2 as cv
import numpy as np
import v4l2 as v4l


def make_multi_trackbar(list_of_property):
    for property in list_of_property:
        cv.createTrackbar(property[0],'Camera_Settings',property[1],property[2],get_value)


def get_value(int):
    value = []
    for property in property_list:
        value.append(cv.getTrackbarPos(property[0],'Camera_Settings'))
    
    print(value
    return value
    
        
def callback(x):
    pass



global property_list
cv.namedWindow('Camera_Settings')

# # command to be passed and the result we expect 
# p = subprocess.Popen(['pip', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# out, err = p.communicate()

# print("\n\n\n\n\n")
# print(out)


# try out new 
# syntax is [property name , default value , max value]

property_list = [["EXPOSURE",0,179],["BRIGHTNESS",0,179],["CONTRAST",0,255],["SATURATION",0,255],["GAIN",0,255],["FPS",0,13]]


property_list = [["expo", 50, 100] , ["bright", 50, 100] , ["contrast", 50, 100] , ["saturation", 50, 100] , ["gain", 50, 100] , ["fps", 50, 100]]

make_multi_trackbar(property_list)
# make_multi_trackbar([["expo1", 70, 100]])

print(get_value(property_list))

cap = cv.VideoCapture(0)
cap.set()


cv.waitKey(0)