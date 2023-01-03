import cv2 as cv
from matplotlib import image
import dev_Tools as dt
from skimage.filters import unsharp_mask
import numpy as np


image_path = r'D:\datasets\scanner bin\contaminated_grainframe78.png'
image = cv.imread(image_path)
img  = cv.imread(image_path)
print(img.shape)
img = dt.background_remover(image_path, lowH = 89, highH = 124, lowS = 149, highS = 255, lowV = 147, highV = 255)

img_2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img_2.shape)

img =  unsharp_mask(img_2, radius=5, amount=2) 
img = img * 255

img = img.astype(np.uint8)
# cv.imwrite(r'0010100.png', img)

print(img.shape)

# img = cv.imread(r'0010100.png')

# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cont, her = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
count =sorted(cont, key=cv.contourArea, reverse=True)

print(len(cont))

# for i in range(len(cont)):
#     color = (np.random.choice(range(256), size=3))
#     color = tuple(color)
#     x,y,z = color

#     # cv.drawContours(image, count[i], -1, (0,255,0), 1)
#     cv.drawContours(image, count[i], -1, (int(x),int(y),int(z)), 2)
# # img = cv.Canny(img, 100, 200)
#     cv.imshow('img', image)
#     cv.waitKey(400)

for i in count:
    for j in i:
        x,y = j.ravel()
        cv.circle(image,(x,y),10,(0,0,255),1)
        cv.imshow("window",image)
        cv.waitKey(100)
cv.waitKey(0)