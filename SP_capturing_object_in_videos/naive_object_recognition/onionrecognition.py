import cv2 as cv
import numpy as np

url = r"https://192.168.29.61:8080/video"
cap = cv.VideoCapture(url)
while(True):
    camera, frame = cap.read()
    if frame is not None:
        img = frame
        # cv.imshow( 'im`',img)
        # converitn all the images in hsv format and bluring a litte
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # hsv = cv.GaussianBlur(hsv_og,(11,11),0)
        # cv.imshow("img",img)

        # defining color values

        # Red
        lower = np.array([25,0,0])
        upper = np.array([144, 255, 255])

        # createing a mask 
        mask = cv.inRange(hsv, lower, upper)
        mask = cv.bitwise_not(mask)
        img = cv.bitwise_and(img , img, mask= mask)
        img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lot_blur = cv.GaussianBlur(img1, (11,11),0)
        # cv.imshow("mask", lot_blur)


        cont, her = cv.findContours(lot_blur, cv.RECURS_FILTER, cv.CHAIN_APPROX_SIMPLE)

        i = 0
        for cont in cont:
            area = cv.contourArea(cont)
            if area > 1000:
                # cv.drawContours(img, cont, -1, (0, 255, 0), 1)
                x,y,h,w = cv.boundingRect(cont)
                cropped_img = img[y:y+w, x:x+h]
                # cv.imshow("apple_obj",cropped_img)
                # name = "apple_obj" + str(i) + ".jpg"
                # cv.imwrite(name, cropped_img)
                img = cv.rectangle(frame, (x,y), (x+h, y+w), (0,255,0), 2)
                i += 1
            cv.imshow("Frame", img)
    q = cv.waitKey(10)
    if q==ord("q"):
        break
cv.destroyAllWindows()






















