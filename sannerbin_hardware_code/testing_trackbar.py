from venv import create
import cv2 as cv
import numpy as np


def brightness(x):
    cap.set(cv.CAP_PROP_SATURATION, x)

while True:
    cap = cv.VideoCapture()
    cap.open(0)

    cv.namedWindow('Camera_Settings')

    cv.createTrackbar('BRIGHTNESS','Camera_Settings',-5,50,brightness)




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
        break 
    cv.waitKey(0)