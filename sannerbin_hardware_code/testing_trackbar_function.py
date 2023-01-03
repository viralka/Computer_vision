import cv2 as cv
import numpy as np

def SATURATION(x):
    cap.set(cv.CAP_PROP_SATURATION, x-100)

def hue(x):
    cap.set(cv.CAP_PROP_HUE, x)

def brightness(x):
    cap.set(cv.CAP_PROP_BRIGHTNESS, x-200)

def CONTRAST(x):
    cap.set(cv.CAP_PROP_CONTRAST, x)

def GAIN(x):
    cap.set(cv.CAP_PROP_GAIN, x)

def FPS(x):
    cap.set(cv.CAP_PROP_FPS, x)

def FRAME_WIDTH(x):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, x)

def FRAME_HEIGHT(x):
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, x)

def AUTO_EXPOSURE():
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)


cap = cv.VideoCapture(0)

cv.namedWindow('Camera_Settings')

cv.createTrackbar('SATURATION','Camera_Settings',150,200,SATURATION)
cv.createTrackbar('HUE','Camera_Settings',0,179,hue)
cv.createTrackbar('BRIGHTNESS','Camera_Settings',200,400,brightness)
cv.createTrackbar('CONTRAST','Camera_Settings',0,100,CONTRAST)
cv.createTrackbar('GAIN','Camera_Settings',0,100,GAIN)
cv.createTrackbar('FPS','Camera_Settings',13,100,FPS)


cv.createTrackbar('FRAME_WIDTH','Camera_Settings',640,2000,FRAME_WIDTH)   
cv.createTrackbar('FRAME_WIDTH','Camera_Settings',640,2000,FRAME_WIDTH)   
cv.createTrackbar('FRAME_WIDTH','Camera_Settings',480,2000,FRAME_HEIGHT)   






catagoty = 'null'
i = 0
while(True):
    camera, frame = cap.read()
    if frame is not None:
        cv.imshow("Frame", frame)
    q = cv.waitKey(1)
    if q==ord("q"):
        break
    elif q==ord("s"):
        name = catagoty+ "frame" + str(i) + ".jpg"
        cv.imwrite(name, frame)
        print("Saved")
        i += 1
    elif q==ord("c"):
        catagoty = input("Enter catagory: ")
    

cv.waitKey(0)