import numpy as np
import cv2 as cv
import glob
import imutils
import _dev_Tools as dt

image_paths = glob.glob('rotating_vid_img/*.jpg')
images = []


i = 0

for i in range(len(image_paths)):
    images.append(cv.imread(image_paths[i*40]))
    
    if i==4:
        break


images.reverse()


# for image in image_paths:
#     img = cv.imread(image)
#     # img = dt.resizeimg(img, 800)
#     images.append(img)

#     if i==4:
#         break
#     i += 1
    
    # cv.imshow("Image", img)
    # cv.waitKey(0)


imageStitcher = cv.Stitcher_create(mode= cv.Stitcher_PANORAMA)

error, stitched_img = imageStitcher.stitch(images)

if not error:

    cv.imwrite("stitchedOutput2.png", stitched_img)
    # cv.imshow("Stitched Img", stitched_img)
    # cv.waitKey(0)



    stitched_img = cv.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv.BORDER_CONSTANT, (0,0,0))

    gray = cv.cvtColor(stitched_img, cv.COLOR_BGR2GRAY)
    thresh_img = cv.threshold(gray, 0, 255 , cv.THRESH_BINARY)[1]

    contours = cv.findContours(thresh_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv.boundingRect(areaOI)
    cv.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()

    while cv.countNonZero(sub) > 0:
        minRectangle = cv.erode(minRectangle, None)
        sub = cv.subtract(minRectangle, thresh_img)


    contours = cv.findContours(minRectangle.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv.contourArea)

    x, y, w, h = cv.boundingRect(areaOI)

    stitched_img = stitched_img[y:y + h, x:x + w]

    cv.imwrite("stitchedOutputProcessed2.png", stitched_img)

    cv.imshow("Stitched Image Processed", stitched_img)

    cv.waitKey(0)



else:
    print("Error: {}".format(error))
    print("Images could not be stitched!")
    print("Likely not enough keypoints being detected!")