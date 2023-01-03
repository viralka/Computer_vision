import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, color, io 

img = cv.imread(r'Main_project\test_data\obj37_0.jpg')
img = img[:,:,2]

from skimage.segmentation import clear_border
opening = clear_border(img)
thres, img = cv.threshold(opening, 0, 255, cv.THRESH_BINARY)


cv.imshow("window",img)
cv.waitKey(0)