import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    #captureing frame
    _, frame_og = cap.read()
    
   
    #blured frame
    frame =  cv.GaussianBlur(frame_og, (5,5), 0)

    # converted BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Defining the color
    lower_blue = np.array([36,8,0]) # failed attempt to find a banan
    upper_blue = np.array([44,29,98])

    # Red
    lower_r = np.array([0, 100, 20])
    upper_r = np.array([10, 255, 255])


    # creating a mask
    mask = cv.inRange(hsv,lower_r, upper_r)
    cv.imshow("mask",mask)

    # # Finding contour
    # _,contour = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(mask , cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # print(contours)
    
    cv.imshow("frame",frame)

    key = cv.waitKey(100)
    if key == 27 :
        break

cap.release()
cv.destroyAllWindows()