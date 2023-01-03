"""
Have some Error  in the code DO not give any conture to any thing 

"""


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import image_opener as io

img_list = io.allimg(r'apple_masking\apple_image')

green = [130,158,0]

scalePercent = 1

diff = 20

boundaries = [([green[2], green[1]-diff, green[0]-diff],
           [green[2]+diff, green[1]+diff, green[0]+diff])]

for img in img_list:
    for (lower, upper) in boundaries:

        # get lower and upper part of the interval

        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        mask = cv.inRange(img, lower, upper)

        # Check out the binary mask:
        cv.imshow("binary mask", mask)
        cv.waitKey(0)

        # Now, you AND the mask and the input image
        # All the pixels that are white in the mask will
        # survive the AND operation, all the black pixels
        # will remain black
        output = cv.bitwise_and(img, img, mask=mask)

        # Check out the ANDed mask:
        cv.imshow("ANDed mask", output)
        cv.waitKey(0)

        # You can use the mask to count the number of white pixels.
        # Remember that the white pixels in the mask are those that
        # fall in your defined range, that is, every white pixel corresponds
        # to a green pixel. Divide by the image size and you got the
        # percentage of green pixels in the original image:
        ratio_green = cv.countNonZero(mask)/(img.size/3)

        # This is the color percent calculation, considering the resize I did earlier.
        colorPercent = (ratio_green * 100) / scalePercent

        # Print the color percent, use 2 figures past the decimal point
        print('green pixel percentage:', np.round(colorPercent, 2))

        # numpy's hstack is used to stack two images horizontally,
        # so you see the various images generated in one figure:
        cv.imshow("images", np.hstack([img, output]))
        cv.waitKey(0)