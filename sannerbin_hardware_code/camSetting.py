import cv2 as cv 
import numpy as np


# supporting function 
def callback(x):
    

    pass




# Creating the windows 
cv.namedWindow('Camera_Settings')

# Default Values

EXPOSURE = 0
BRIGHTNESS = 50
CONTRAST = 0
SATURATION = 0
GAIN = 0
FPS = 13

# create trackbars for color change
cv.createTrackbar('EXPOSURE','Camera_Settings',EXPOSURE,179,callback)
cv.createTrackbar('BRIGHTNESS','Camera_Settings',BRIGHTNESS,179,callback)
cv.createTrackbar('CONTRAST','Camera_Settings',CONTRAST,255,callback)
cv.createTrackbar('SATURATION','Camera_Settings',SATURATION,255,callback)
cv.createTrackbar('GAIN','Camera_Settings',GAIN,255,callback)
cv.createTrackbar('FPS','Camera_Settings',FPS,13,callback)

psum = 0
while True:

    # gettin the variabel 

    EXPOSURE = cv.getTrackbarPos("EXPOSURE", "Camera_Settings")
    BRIGHTNESS = cv.getTrackbarPos("BRIGHTNESS", "Camera_Settings")
    CONTRAST = cv.getTrackbarPos("CONTRAST", "Camera_Settings")
    SATURATION = cv.getTrackbarPos("SATURATION", "Camera_Settings")
    GAIN = cv.getTrackbarPos("GAIN", "Camera_Settings")
    FPS = cv.getTrackbarPos("FPS", "Camera_Settings")







    # logic for camera settings
    sum = EXPOSURE + BRIGHTNESS + CONTRAST + SATURATION + GAIN + FPS

    cap = cv.VideoCapture(0)

    if sum - psum != 0:
        print("change")
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv.CAP_PROP_EXPOSURE, EXPOSURE)
        cap.set(cv.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
        cap.set(cv.CAP_PROP_CONTRAST, CONTRAST)
        cap.set(cv.CAP_PROP_SATURATION, SATURATION)
        cap.set(cv.CAP_PROP_GAIN, GAIN)
        cap.set(cv.CAP_PROP_FPS, FPS)
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        # cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    
    psum = sum


    # reading the camera frame 

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





        
        