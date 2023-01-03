import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import image_opener as io

# opened image 
# img_list = io.allimg(r'apple_masking\apple_image')
# img = img_list[8]
img = cv.imread(r"apple_masking\apple_image\ad2.jpg")
# img = cv.resize(img,(450,672))
cv.imshow("img",img)

# finding contoure using lapacian of gaussian for edge detection
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
prams = cv.SimpleBlobDetector_Params()

# # filtering data
# prams.minThreshold = 0
# prams.maxThreshold = 225

#filter by color values
prams.filterByColor = 1
prams.blobColor = 0

# filtering by circularity values
prams.filterByCircularity = True
prams.minCircularity = 0
prams.maxCircularity = 10

# filtering by inertialRations
prams.filterByInertia = True
prams.minInertiaRatio = 0.1
prams.maxInertiaRatio = 0.9

#filter by area values
prams.filterByArea = True
prams.minArea = 50
prams.maxArea = 1000

# filter by distance between the blobls
prams.minDistBetweenBlobs = 0

# set up the thing
detector = cv.SimpleBlobDetector_create(prams)

# Detect blobs
Keypoints = detector.detect(img)

print("Number of blobs found: ", len(Keypoints))


# Draw blobs
img_with_blobs = cv.drawKeypoints(img, Keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_with_blobs)
cv.imshow("img_with_blobs", img_with_blobs)
cv.waitKey(0)
cv.destroyAllWindows()


