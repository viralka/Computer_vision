import cv2 as cv
import numpy as np
# opening all the images from a folder
img = cv.imread(r'C:\Users\mridu\Desktop\agnext\Agnext\OpenCv\apple_masking\apple_image\afaff.jpg')
img = cv.resize(img,(500,500))
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower_blue = np.array([70, 0, 0])
upper_blue = np.array([145, 255, 255])
mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
mask_blue = cv.bitwise_not(mask_blue)
img = cv.bitwise_and(img, img, mask=mask_blue)
cv.imshow("img", img)
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
        name = "apple_obj" + str(i) + ".jpg"
        cv.imwrite(name, cropped_img)
        img = cv.rectangle(img, (x,y), (x+h, y+w), (0,255,0), 2)
        i += 1
cv.waitKey(00)