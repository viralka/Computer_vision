import numpy as np
import cv2 as cv
import glob
import imutils
import _dev_Tools as dt

image_paths = glob.glob('rotating_vid_img/*.jpg')
images = []

  
pimag = img = cv.imread(image_paths[0])[:,800:801]   

i=0
while True:
    
 
    image_paths =    r'D:\agnext\Agnext\OpenCv\_Image_stiching\rotating_vid_img\frame'+str(i)+r'.jpg'
    # print(image_paths)
    # break
    try:
        img2 = cv.imread(image_paths)[:,800:801] 
    except:
        break
    # print(image_paths)

    img1 = pimag
    test = np.concatenate((img1,img2), axis=1)
    # cv.imshow("Image", test)
    # cv.waitKey(0)
    pimag = test
    i += 1
 

cv.imwrite("apple_test_13.png", test)

