import cv2 as cv
import numpy as np
import image_opener as ip


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