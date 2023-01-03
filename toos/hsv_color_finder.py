import cv2
import numpy as np
import image_opener as io

def callback(x):
    pass

# cap = cv2.VideoCapture(0)
# img = cv2.imread('IMG_20201224_12304.jpg')


#img = cv2.imread('wheat-upper1.png')
img = cv2.imread(r"D:\datasets\AutoSampler\frames\Frame46.jpg")
img = io.resizeimg(img, 1000)
new_rows = int(img.shape[0]/2)
new_cols = int(img.shape[1]/2)
img = cv2.resize(img, (new_cols,new_rows), fx=0.5, fy=0.5)

cv2.namedWindow('Colorbars1')
# cv2.namedWindow('Colorbars2')
# cv2.namedWindow('Colorbars3')

ilowH = 0
ihighH = 179

ilowS = 0
ihighS = 255
ilowV = 0
ihighV = 255



# create trackbars for color change
cv2.createTrackbar('lowH','Colorbars1',ilowH,179,callback)
cv2.createTrackbar('highH','Colorbars1',ihighH,179,callback)

cv2.createTrackbar('lowS','Colorbars1',ilowS,255,callback)
cv2.createTrackbar('highS','Colorbars1',ihighS,255,callback)

cv2.createTrackbar('lowV','Colorbars1',ilowV,255,callback)
cv2.createTrackbar('highV','Colorbars1',ihighV,255,callback)

# create trackbars for color change
# cv2.createTrackbar('lowH','Colorbars2',ilowH,179,callback)
# cv2.createTrackbar('highH','Colorbars2',ihighH,179,callback)

# cv2.createTrackbar('lowS','Colorbars2',ilowS,255,callback)
# cv2.createTrackbar('highS','Colorbars2',ihighS,255,callback)

# cv2.createTrackbar('lowV','Colorbars2',ilowV,255,callback)
# cv2.createTrackbar('highV','Colorbars2',ihighV,255,callback)


# # create trackbars for color change
# cv2.createTrackbar('lowH','Colorbars3',ilowH,179,callback)
# cv2.createTrackbar('highH','Colorbars3',ihighH,179,callback)

# cv2.createTrackbar('lowS','Colorbars3',ilowS,255,callback)
# cv2.createTrackbar('highS','Colorbars3',ihighS,255,callback)

# cv2.createTrackbar('lowV','Colorbars3',ilowV,255,callback)
# cv2.createTrackbar('highV','Colorbars3',ihighV,255,callback)


img2 = np.copy(img)
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
while(1):
    # ret, frame = cap.read()
    ilowH1 = cv2.getTrackbarPos("lowH", "Colorbars1")
    ihighH1 = cv2.getTrackbarPos("highH", "Colorbars1")
    ilowS1 = cv2.getTrackbarPos("lowS", "Colorbars1")
    ihighS1 = cv2.getTrackbarPos("highS", "Colorbars1")
    ilowV1 = cv2.getTrackbarPos("lowV", "Colorbars1")
    ihighV1 = cv2.getTrackbarPos("highV", "Colorbars1")

    
    # # ret, frame = cap.read()
    # ilowH2 = cv2.getTrackbarPos("lowH", "Colorbars2")
    # ihighH2 = cv2.getTrackbarPos("highH", "Colorbars2")
    # ilowS2 = cv2.getTrackbarPos("lowS", "Colorbars2")
    # ihighS2 = cv2.getTrackbarPos("highS", "Colorbars2")
    # ilowV2 = cv2.getTrackbarPos("lowV", "Colorbars2")
    # ihighV2 = cv2.getTrackbarPos("highV", "Colorbars2")

    
    # # ret, frame = cap.read()
    # ilowH3 = cv2.getTrackbarPos("lowH", "Colorbars3")
    # ihighH3 = cv2.getTrackbarPos("highH", "Colorbars3")
    # ilowS3 = cv2.getTrackbarPos("lowS", "Colorbars3")
    # ihighS3 = cv2.getTrackbarPos("highS", "Colorbars3")
    # ilowV3 = cv2.getTrackbarPos("lowV", "Colorbars3")
    # ihighV3 = cv2.getTrackbarPos("highV", "Colorbars3")

    
    
    lower_hsv1 = np.array([ilowH1, ilowS1, ilowV1])
    higher_hsv1 = np.array([ihighH1, ihighS1, ihighV1])
    mask1 = cv2.inRange(hsv, lower_hsv1, higher_hsv1)
    cv2.imshow('mask1', mask1)
    
    
    # lower_hsv2 = np.array([ilowH2, ilowS2, ilowV2])
    # higher_hsv2 = np.array([ihighH2, ihighS2, ihighV2])
    # mask2 = cv2.inRange(hsv, lower_hsv2, higher_hsv2)
    # cv2.imshow('mask2', mask2)
    
    
    # lower_hsv3 = np.array([ilowH3, ilowS3, ilowV3])
    # higher_hsv3 = np.array([ihighH3, ihighS3, ihighV3])
    # mask3 = cv2.inRange(hsv, lower_hsv3, higher_hsv3)
    # cv2.imshow('mask3', mask3)

    
    
    
    # maskf1 = cv2.bitwise_or(mask1, mask2)
    # mask = cv2.bitwise_or(maskf1, mask3)
    
    mask = mask1

    img2 = np.copy(img)
    img2[mask == 0] = 0

    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#    hsv2[:,:,2] = gray

    

    cv2.imshow('frame', img2)
#    cv2.imshow('hsv', hsv2)

#    cv2.imshow('gray', gray)
    # print(ilowH, ihighH,  ilowS, ihighS,  ilowV, ihighV )

    
    if(cv2.waitKey(1) & 0xFF == 27):# ord('q')):
        break


cv2.destroyAllWindows()



