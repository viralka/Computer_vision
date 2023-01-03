from re import M
from anyio import connect_unix
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import meijering
from skimage.measure import find_contours
import image_opener as ip
imgo = cv.imread(r"Main_project\test_data\5.jpg")
img = ip.resizeimg(imgo, 700)
imggr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imggr = cv.GaussianBlur(imggr, (7, 7), 0)
# imggr = cv.adaptiveThreshold(imggr, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# imggr = cv.dilate(imggr, None, iterations=4)

cv.imshow("window",img)

# # meijering 
meijering_color = meijering(imggr, sigmas=range(1, 10, 2), alpha=None, black_ridges=True, mode='reflect', cval=0)


cv.imshow("window",meijering_color)

# meijering_color = meijering_color[:,:,0]


count = find_contours(meijering_color, 0.17)
# count, _ = cv.findContours(meijering_color, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# # Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(imggr, cmap=plt.cm.gray)

for contour in count:
    if  len(contour) > 500:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
cv.drawContours(img, count, -1, (0,255,0), 3)


cv.imshow("contours",img)



cv.waitKey(0)