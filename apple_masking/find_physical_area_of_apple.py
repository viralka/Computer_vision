import cv2 as cv
import numpy as np
import image_opener as ip

def area_in(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blank = np.zeros(img.shape, dtype='uint8')
    blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
    contours, hierarchies = cv.findContours(blur, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours, key=cv.contourArea, reverse= True)
    return cv.contourArea(contours[0])
    

# opening all the images from a folder
# img_list = ip.allimg(r'apple_masking\apple_image')
# img = img_list[6]
img = cv.imread(r'apple_masking\apple_image\afaff.jpg')
img = ip.resizeimg(img, 500)
# cv.imshow( 'im`',img)


# converitn all the images in hsv format and bluring a litte
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# hsv = cv.GaussianBlur(hsv_og,(11,11),0)
# cv.imshow("img",img)

# defining color values

# background
lower = np.array([15,0,0])
upper = np.array([116, 255, 255])

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
        if i==1:
            cropped_img = img[y:y+w, x:x+h]
            apple = cropped_img
            # cv.imshow("apple_obj",cropped_img)
            name = "apple_obj" + str(i) + ".jpg"
            cv.imwrite(name, cropped_img)
            apple = cv.rectangle(img, (x,y), (x+h, y+w), (0,255,0), 2)
            area_by_length_bar.append(h*w)
            i += 1
        else:
            cropped_img = img[y:y+w, x:x+h]
            bars.append(cropped_img)
            # cv.imshow("apple_obj",cropped_img)
            name = "side_bar" + str(i) + ".jpg"
            cv.imwrite(name, cropped_img)
            img = cv.rectangle(img, (x,y), (x+h, y+w), (0,255,0), 2)
            area_by_length_apple.append(h*w)
            i += 1

# now the area of side bar is known to be 12 x 3 cm

# done by conture area
areas =[]

for img in bars:
    area1 = area_in(img)
    print(area1)
    areas.append(area1)

average_area = max(areas)
conversion_factor = 36/average_area
physical_area_apple= conversion_factor*area_in(apple)

area_of_apple = area_in(apple)
print("Physical area of Apple is: ", round(physical_area_apple,3), " cm^2 (by conture area)")

# done by length area
average_area = max(area_by_length_bar)
conversion_factor = 36/average_area
physical_area_apple= 2*conversion_factor*max(area_by_length_apple)

area_of_apple = area_in(apple)
print("Physical area of Apple is: ", round(physical_area_apple,3), " cm^2 (by leangth  area)")

cv.waitKey(00)