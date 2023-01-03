"""
can not find are of the specifc conture 

"""


import cv2
import numpy as np

# load image
img = cv2.imread(r'apple_masking\apple_image\a4.jpg')

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

# create mask for red color in hsv

lower = np.array([0, 100, 20])
upper = np.array([10, 255, 255])

# create mask for blue color in hsv

mask = cv2.inRange(hsv, lower, upper)

# count non-zero pixels in mask


count=np.count_nonzero(mask)
print('count:', count)

# save output
cv2.imwrite('blue_bag_mask.png', mask)

# Display various images to see the steps
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()



