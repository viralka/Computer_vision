import cv2 as cv
from matplotlib import image
import numpy as np
import os
from re import I
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt



# resizing images but also saving it ratio
def resizeimg(img, size, interpolation = cv.INTER_CUBIC):
    # resizing images but also saving it ratio
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv.resize(mask, (size, size), interpolation)

#open all the images in a folder
def allimg(folder):
    #open all the images in a folder
    image =load_images_from_folder(folder)
    imlist=[]
    i = 0
    for img in image:
        name = "img" + str(i)
        imlist.append(img)
    
    return imlist
    
# load_images_from_folder
def load_images_from_folder(folder):
    # load_images_from_folder
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# hsv finder for the color 
def callback(x):

    pass

def hsv_finder(image_path):
    # hsv finder for the color 
    img = cv.imread(image_path)
    img = resizeimg(img, 1500)
    new_rows = int(img.shape[0]/2)
    new_cols = int(img.shape[1]/2)
    img = cv.resize(img, (new_cols,new_rows), fx=0.5, fy=0.5)

    cv.namedWindow('Colorbars1')
    # cv.namedWindow('Colorbars2')
    # cv.namedWindow('Colorbars3')

    ilowH = 0
    ihighH = 179

    ilowS = 0
    ihighS = 255
    ilowV = 0
    ihighV = 255



    # create trackbars for color change
    cv.createTrackbar('lowH','Colorbars1',ilowH,179,callback)
    cv.createTrackbar('highH','Colorbars1',ihighH,179,callback)

    cv.createTrackbar('lowS','Colorbars1',ilowS,255,callback)
    cv.createTrackbar('highS','Colorbars1',ihighS,255,callback)

    cv.createTrackbar('lowV','Colorbars1',ilowV,255,callback)
    cv.createTrackbar('highV','Colorbars1',ihighV,255,callback)

    # create trackbars for color change
    # cv.createTrackbar('lowH','Colorbars2',ilowH,179,callback)
    # cv.createTrackbar('highH','Colorbars2',ihighH,179,callback)

    # cv.createTrackbar('lowS','Colorbars2',ilowS,255,callback)
    # cv.createTrackbar('highS','Colorbars2',ihighS,255,callback)

    # cv.createTrackbar('lowV','Colorbars2',ilowV,255,callback)
    # cv.createTrackbar('highV','Colorbars2',ihighV,255,callback)


    # # create trackbars for color change
    # cv.createTrackbar('lowH','Colorbars3',ilowH,179,callback)
    # cv.createTrackbar('highH','Colorbars3',ihighH,179,callback)

    # cv.createTrackbar('lowS','Colorbars3',ilowS,255,callback)
    # cv.createTrackbar('highS','Colorbars3',ihighS,255,callback)

    # cv.createTrackbar('lowV','Colorbars3',ilowV,255,callback)
    # cv.createTrackbar('highV','Colorbars3',ihighV,255,callback)


    img2 = np.copy(img)
    hsv = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
    while(1):
        # ret, frame = cap.read()
        ilowH1 = cv.getTrackbarPos("lowH", "Colorbars1")
        ihighH1 = cv.getTrackbarPos("highH", "Colorbars1")
        ilowS1 = cv.getTrackbarPos("lowS", "Colorbars1")
        ihighS1 = cv.getTrackbarPos("highS", "Colorbars1")
        ilowV1 = cv.getTrackbarPos("lowV", "Colorbars1")
        ihighV1 = cv.getTrackbarPos("highV", "Colorbars1")


        # # ret, frame = cap.read()
        # ilowH2 = cv.getTrackbarPos("lowH", "Colorbars2")
        # ihighH2 = cv.getTrackbarPos("highH", "Colorbars2")
        # ilowS2 = cv.getTrackbarPos("lowS", "Colorbars2")
        # ihighS2 = cv.getTrackbarPos("highS", "Colorbars2")
        # ilowV2 = cv.getTrackbarPos("lowV", "Colorbars2")
        # ihighV2 = cv.getTrackbarPos("highV", "Colorbars2")


        # # ret, frame = cap.read()
        # ilowH3 = cv.getTrackbarPos("lowH", "Colorbars3")
        # ihighH3 = cv.getTrackbarPos("highH", "Colorbars3")
        # ilowS3 = cv.getTrackbarPos("lowS", "Colorbars3")
        # ihighS3 = cv.getTrackbarPos("highS", "Colorbars3")
        # ilowV3 = cv.getTrackbarPos("lowV", "Colorbars3")
        # ihighV3 = cv.getTrackbarPos("highV", "Colorbars3")



        lower_hsv1 = np.array([ilowH1, ilowS1, ilowV1])
        higher_hsv1 = np.array([ihighH1, ihighS1, ihighV1])
        mask1 = cv.inRange(hsv, lower_hsv1, higher_hsv1)
        cv.imshow('mask1', mask1)


        # lower_hsv2 = np.array([ilowH2, ilowS2, ilowV2])
        # higher_hsv2 = np.array([ihighH2, ihighS2, ihighV2])
        # mask2 = cv.inRange(hsv, lower_hsv2, higher_hsv2)
        # cv.imshow('mask2', mask2)


        # lower_hsv3 = np.array([ilowH3, ilowS3, ilowV3])
        # higher_hsv3 = np.array([ihighH3, ihighS3, ihighV3])
        # mask3 = cv.inRange(hsv, lower_hsv3, higher_hsv3)
        # cv.imshow('mask3', mask3)




        # maskf1 = cv.bitwise_or(mask1, mask2)
        # mask = cv.bitwise_or(maskf1, mask3)

        mask = mask1

        img2 = np.copy(img)
        img2[mask == 0] = 0

        hsv2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)

        gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    #    hsv2[:,:,2] = gray



        cv.imshow('frame', img2)
    #    cv.imshow('hsv', hsv2)

    #    cv.imshow('gray', gray)
        # print(ilowH, ihighH,  ilowS, ihighS,  ilowV, ihighV )


        if(cv.waitKey(1) & 0xFF == 27):# ord('q')):
            break


    cv.destroyAllWindows()

# baground remover and object segmentation using color mask
def bg_remover_segmentation(image_path, lowH = 0, highH = 255, lowS = 0, highS = 255, lowV = 0, highV = 255):
    # baground remover and object segmentation using color mask
    # opening all the images from a folder

    img = cv.imread(image_path)
    cv.imshow('original', img)

    # converitn all the images in hsv format and bluring a litte
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # hsv = cv.GaussianBlur(hsv_og,(11,11),0)
    cv.imshow("img",img)

    # defining color values

    # mask value
    lower = np.array([lowH, lowS, lowV])
    upper = np.array([highH, highS, highV])

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
            name = "obj" + str(i) + ".jpg"
            # cv.imwrite(name, cropped_img)
            cv.imshow(name,cropped_img)
            img = cv.rectangle(img, (x,y), (x+h, y+w), (0,255,0), 2)
            i += 1





    img = cv.imshow("img",img)

    cv.waitKey(00)

# physical area finder #! this function might have some issues
def area_finder(image_path, conversion_factor = 1, lowH = 0, highH = 255, lowS = 0, highS = 255, lowV = 0, highV = 255):
    # physical area finder
    def area_in(img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blank = np.zeros(img.shape, dtype='uint8')
        blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
        contours, hierarchies = cv.findContours(blur, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours= sorted(contours, key=cv.contourArea, reverse= True)
        return cv.contourArea(contours[0])
    

    img = cv.imread(image_path)
    img = resizeimg(img, 500)
    # cv.imshow( 'im`',img)


    # converitn all the images in hsv format and bluring a litte
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # hsv = cv.GaussianBlur(hsv_og,(11,11),0)
    # cv.imshow("img",img)

    # defining color values

    # background
    lower = np.array([lowH, lowS, lowV])
    upper = np.array([highH, highS, highV])

    # createing a mask 
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.bitwise_not(mask)
    img = cv.bitwise_and(img , img, mask= mask)
    img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lot_blur = cv.GaussianBlur(img1, (11,11),0)
    # cv.imshow("mask", lot_blur)


    cont, her = cv.findContours(lot_blur, cv.RECURS_FILTER, cv.CHAIN_APPROX_SIMPLE)


    i = 0
    bars=[]
    apple = 0
    area_by_length_bar = []
    area_by_length_apple = []

    for cont in cont:
        area = cv.contourArea(cont)
        if area > 1000:
            # cv.drawContours(img, cont, -1, (0, 255, 0), 1)
            x,y,h,w = cv.boundingRect(cont)
            cropped_img = img[y:y+w, x:x+h]
            apple = cropped_img
            area_by_length_bar.append(h*w)
            i += 1
    

    # now the area of side bar is known to be 12 x 3 cm

    # done by conture area

    physical_area_apple= conversion_factor*area_in(apple)

    area_of_apple = area_in(apple)
    print("Physical area of in color_range is: ", round(physical_area_apple,3), " cm^2 (by conture area)")

    # done by length area
    average_area = max(area_by_length_bar)
    conversion_factor = 36/average_area
    physical_area_apple= 2*conversion_factor*max(area_by_length_apple)

    area_of_apple = area_in(apple)
    print("Physical area of in color_range is: ", round(physical_area_apple,3), " cm^2 (by leangth  area)")

    cv.waitKey(00)

# extrem point finder  #! this function might have some issues
def extrem_point_finder(image_path):
    # extrem point finder
    img = cv.imread(image_path)
    img1 = resizeimg(img, 1000)
    img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    mask = np.zeros(img.shape, dtype=np.uint8)


    # # img = cv.GaussianBlur(img, (1,1),0)
    # img = cv.erode(img, (3,3),iterations=7)
    # img = cv.dilate(img, (3,3),iterations=7)

    top, bottom, left, right, angletb,angle1tb = extreme_points(img, mask)

    print("Top: ", top, " Bottom: ", bottom, " Left: ", left, " Right: ", right, " Angle: ", angletb, " Angle: ", angle1tb)
    
    cv.line(img1 , top, bottom, (0,0,255), 2)
    cv.line(img1 , left, right, (0,0,255), 2)

    cv.imshow('img',img1)
    img = cv.Canny(img, 100, 200)

    count , _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    name = "contours2.jpg"
    cv.imwrite(name, mask)


    cv.waitKey(00)
    

def angle(p1,p2):
    return np.arctan2(p2[1]-p1[1],p2[0]-p1[0])

def extreme_points(img, mask):
    cont, her = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    count = max(cont, key = cv.contourArea)

    cv.drawContours(mask, count, -1, (225, 0 , 0), 1)

    top = tuple(count[count[:,:,1].argmin()][0])
    bottom = tuple(count[count[:,:,1].argmax()][0])
    left = tuple(count[count[:,:,0].argmin()][0])
    right = tuple(count[count[:,:,0].argmax()][0])


    return (top, bottom, left, right,angle(top, bottom),angle(left, right))


# cutting image by the corners
def rcut(img, roi_corne):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    roi_corne = np.array(roi_corne, dtype="int32")
    channel_count = img.shape[2]
    ignore_mask = (255)*channel_count
    cv.fillPoly(mask, [roi_corne], 255)

    masked_image = cv.bitwise_and(img, img, mask=mask)

    return masked_image

def indexf(count, points):
    for i in range(len(count)):
        for j in range(len(count[i])):
            for k in range(len(count[i][j])):
                if count[i][j][k][0] == points[0] and count[i][j][k][1] == points[1]:
                    return (i,j,k)

def slice_img(count, pi, pf):
    list_of_points = []
    for i in range(pi[0], pf[0]+1):
        for j in range(pi[1], pf[1]+1):
            for k in range(pi[2], pf[2]+1):
                list_of_points.append(list(count[i][j][k]))

def rcut(img, roi_corne):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    roi_corne = np.array(roi_corne, dtype="int32")
    channel_count = img.shape[2]
    ignore_mask = (255)*channel_count
    cv.fillPoly(mask, [roi_corne], 255)

    masked_image = cv.bitwise_and(img, img, mask=mask)

    return masked_image


