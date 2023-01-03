import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import image_opener as ip

# boilerplate for every coder
# opening all the images from a folder
img_list = ip.allimg(r'apple_masking\Defenct_detection\bad_apple_img')


# looping over all images 

for img in img_list:
    cv.imshow('img',img)

    # need to be flat to check it 
    img2 = img.reshape(-1,3)
    img2 = np.float32(img2)

    # creating criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)


    # clusters
    k= 4

    # number of attempts
    attempts = 10

    ret, label, center = cv.kmeans(img2, k, None, criteria, attempts, cv.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv.imshow('res2',res2)

 
    print(center)



    cv.waitKey(2200)