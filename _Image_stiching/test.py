import cv2 as cv 
import numpy as np


vid = cv.VideoCapture(r'D:\agnext\Agnext\OpenCv\_Image_stiching\rotating_video\Apple - 65426.mp4')

i = 0
while True:
    cap , frame = vid.read()
    if cap == False:
        break
    # cv.imshow('frame',frame)
    # cv.waitKey(1)
    name = 'frame' + str(i) + '.jpg'
    cv.imwrite(name,frame)
    i += 1





