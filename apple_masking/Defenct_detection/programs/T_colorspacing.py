import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import image_opener as ip

def test(x):
    pass
# opening all the images from a folder
img_list = ip.allimg(r'apple_masking\Defenct_detection\bad_apple_img')


# looping over all images 

found_valiable = {}
i = 0
for img in img_list:
    # cv.imshow('img',img)
    
    # making window
    cv.namedWindow("window")
    cv.resizeWindow("window", 700, 400)

    # creating trackbars

    cv.createTrackbar("L h", "window", 0, 255, test)
    cv.createTrackbar("H h", "window", 255, 255, test)

    cv.createTrackbar("L s", "window", 0, 255, test)
    cv.createTrackbar("H s", "window", 255, 255, test)

    cv.createTrackbar("L v", "window", 0, 255,test)
    cv.createTrackbar("H v", "window", 255, 255,test)

    img2 = np.copy(img)
    hsv = cv.cvtColor(img2, cv.COLOR_BGR2HSV)

    while True:

        k = cv.waitKey(1) & 0xFF
        # print("in loop")
        

        # filtering data
        lh = cv.getTrackbarPos("L h", 'window')
        hh = cv.getTrackbarPos("H h", 'window')

        hs = cv.getTrackbarPos("H s", 'window')
        ls = cv.getTrackbarPos("L s", 'window')

        hv = cv.getTrackbarPos("H v", 'window')
        lv = cv.getTrackbarPos("L v", 'window')

        lower_rot = np.array([lh, ls, lv])
        upper_rot = np.array([hh, hs, hv])
        mask_rot = cv.inRange(hsv, lower_rot, upper_rot)
        cv.imshow("mask", mask_rot)


        # masked_rotten = cv.bitwise_and(img, img, mask=mask_rot)
        img2 = np.copy(img)
        img2[mask_rot == 0] = 0

        hsv2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)

        gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        #    hsv2[:,:,2] = gray

    

        cv.imshow('frame', img2)




        if k == 27:
            found_valiable[i] = ((hh,hs,hv),(lh,ls,lv))
            i += 1
            break


    

    cv.waitKey(2200)
    cv.destroyAllWindows()

print("----------------------------------------------------------------")
print()
print()
print(found_valiable)
print()
print()