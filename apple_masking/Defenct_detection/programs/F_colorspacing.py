import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import image_opener as ip


# opening all the images from a folder
img_list = ip.allimg(r'apple_masking\Defenct_detection\bad_apple_img')


# looping over all images 

for img in img_list:
    # converting to hsv
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Creating mask of rotten part

    lower_rot = np.array([6, 100, 145])
    upper_rot = np.array([23, 190, 210])
    mask_rot = cv.inRange(hsv, lower_rot, upper_rot)
    masked_rotten = cv.bitwise_and(img, img, mask=mask_rot)

    # Finding rotten area 

    masked_rotten = cv.cvtColor(masked_rotten, cv.COLOR_BGR2GRAY)
    masked_rotten = cv.GaussianBlur(masked_rotten, (11, 11), 0)
    cont_rotten, her_rotten = cv.findContours(
        masked_rotten, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cont_rotten = sorted(cont_rotten, key=cv.contourArea, reverse=True)
    area_rotten = 0
    for cont in cont_rotten:
        area_rotten += cv.contourArea(cont)
    cv.drawContours(img, cont_rotten, -1, (0, 0, 255), 2)
    cv.imshow("Image", img)


    # Creating mask of background to find area of background

    lower_bg = np.array([0, 0, 204])
    upper_bg = np.array([179, 65, 255])
    mask_bg = cv.inRange(hsv, lower_bg, upper_bg)
    masked_bg = cv.bitwise_and(img, img, mask=mask_bg)

    # cv.imshow("removedbg", masked_bg)
    masked_bg = cv.cvtColor(masked_bg, cv.COLOR_BGR2GRAY)
    area_bg = cv.countNonZero(masked_bg)


    # Calculations 

    height = int(img.shape[0])
    width = int(img.shape[1])
    total_area = height*width
    area_fruit = total_area - area_bg

    percentage_rotten = round((area_rotten/area_fruit)*100, 2)
    print(f"Percentage of rotten fruit is {percentage_rotten} %")
    if (percentage_rotten > 5):
        print("rotten")
    else:
        print("good")

    cv.waitKey(2200)
    cv.destroyAllWindows()