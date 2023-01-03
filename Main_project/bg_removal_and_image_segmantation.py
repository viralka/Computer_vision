import cv2 as cv
import numpy as np
import image_opener as ip
import os 

def findEndpoints(skeleton):

	(rows,cols) = np.nonzero(skeleton)
	# Initialize empty list of co-ordinates
	skel_coords = []

	for (r,c) in zip(rows,cols):
		counter = countNeighbouringPixels(skeleton, r,c)
		if counter == 1:
			skel_coords.append((r,c))
	return skel_coords

def drawPoints(points,shape):

	rgb_img = np.float32(shape)
	# print(rgb_img)
	rgb_img = cv.cvtColor(rgb_img,cv.COLOR_GRAY2RGB)
	colour = (255,0,0)
	for i in points:
		rgb_img[i[0]][i[1]] = colour
		# cv.circle(backtorgb, (i[1],i[0]),  3, colour, -1)
	# print(i)
	return rgb_img

def findIntersections(skeleton):

	(rows,cols) = np.nonzero(skeleton)

	# Initialize empty list of co-ordinates
	skel_coords = []

	for (r,c) in zip(rows,cols):
		counter = countNeighbouringPixels(skeleton, r,c)
		if counter >= 4:
			skel_coords.append((r,c))
	return skel_coords

def countNeighbouringPixels(skeleton,x,y):
	# get neighbours
	neighbours = Neighbours(x,y,skeleton)
	return sum(neighbours)/255

def countSkeletonPixels(skeleton):	

	pixels = 0
	for i in range(0, len(skeleton)):
		for j in range(0, len(skeleton[0])):
			if skeleton[i][j] != 0:
				pixels += 1
	return pixels

def Neighbours(x,y,img):
    if x == 0:
        if y == 0:
            return [0,0,img[x][y+1],img[x+1][y]]
        elif y == len(img[0])-1:
            return [img[x][y-1],0,0,img[x+1][y]]
        else:
            return [img[x][y-1],img[x][y+1],img[x+1][y],0]
    elif x == len(img)-1:
        if y == 0:
            return [img[x][y+1],0,0,img[x-1][y]]
        elif y == len(img[0])-1:
            return [img[x][y-1],img[x-1][y],0,0]
        else:
            return [img[x][y-1],img[x][y+1],img[x-1][y],0]
    else:    
        if y == 0:
            return [img[x][y+1],img[x-1][y],img[x+1][y],0]
        elif y == len(img[0])-1:
            return [img[x][y-1],img[x-1][y],img[x+1][y],0]
        else:
            return [img[x][y-1],img[x][y+1],img[x-1][y],img[x+1][y]]




# opening all the images from a folder
img_list = ip.allimg(r'D:\agnext\Agnext\OpenCv\Main_project\dataset')


for img in img_list:

    # converitn all the images in hsv format and bluring a litte
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # hsv = cv.GaussianBlur(hsv_og,(11,11),0)
    # cv.imshow("img",img)

    # defining color values

    # Red
    lower = np.array([0,0,145])
    upper = np.array([255, 255, 255])

    # createing a mask 
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.bitwise_not(mask)
    
    # maskFor_slke = cv.GaussianBlur(mask,(11,11),0)
    # maskFor_slke = cv.erode(maskFor_slke,(5,5) ,iterations=4)
    # maskFor_slke = cv.dilate(maskFor_slke,(5,5),iterations=4)
    # maskFor_slke = cv.erode(maskFor_slke,(5,5) ,iterations=4)
    # maskFor_slke = cv.dilate(maskFor_slke,(5,5),iterations=4)
    # cv.imshow("mask",maskFor_slke)

    # thinned = cv.ximgproc.thinning(maskFor_slke) # making skeleton
    # cv.imshow("skel",thinned)

    # endpontlist = findIntersections(thinned)
    # print(endpontlist)

    # for point in endpontlist:
    #     cv.circle(img, (point[1],point[0]),  3, (0,0,255), -1)

    
    img = cv.bitwise_and(img , img, mask= mask)
    img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # lot_blur = cv.GaussianBlur(img1, (1,1),0)
    lot_blur = img1
    # cv.imshow("mask", lot_blur)


    cont, her = cv.findContours(lot_blur, cv.RECURS_FILTER, cv.CHAIN_APPROX_SIMPLE)

    j=0
    i = 0
    for cont in cont:
        area = cv.contourArea(cont)
        if area > 400 and area < 5000:
            # cv.drawContours(img, cont, -1, (0, 255, 0), 1)
            x,y,h,w = cv.boundingRect(cont)
            cropped_img = img[y:y+w, x:x+h]
            # cv.imshow("apple_obj",cropped_img)
            name = "rice_obj" + str(j)+"_"+str(i) + ".jpg"
            path = 'D:\datasets\rice footage\object_in_the_frame'
            # cv.imwrite(os.path.join(path , name),cropped_img)
            # print(os.path.join(path , name
            cv.imwrite(name, cropped_img)
            print(name)
            # img = cv.rectangle(img, (x,y), (x+h, y+w), (0,255,0), 2)
            i += 1
        j +=1
    




    # img = cv.imshow("img",img)

