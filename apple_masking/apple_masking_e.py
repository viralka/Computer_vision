import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import image_opener as io

# img_list = io.allimg(r'apple_masking\apple_image')
img_list = [cv.imread(r"apple_masking\apple_image\1.png")]

# colored histogram

for img in img_list:

    plt.figure()
    plt.title("Histogram for colored image")
    plt.xlabel("Number of pixels")
    plt.ylabel("Brightness of pixels")
    totalpixels = np.sum( img >0)

    color = ('b','r','g')
    for i,col in enumerate(color):
        cv.imshow("image with color", img)
        hist = cv.calcHist([img],[i], None, [256], [0,256])
        plt.plot(hist, color = col)
        plt.xlim([0, 256])
    
    b,g,r = cv.split(img)
    z = np.zeros(img.shape[:2], dtype = "uint8" )

    # blue = cv.merge([b,z,z])
    blue = b
    num_of_blue = np.sum( blue >0)
    print("num_of_blue pixels: ", num_of_blue)
    percent = (num_of_blue/totalpixels)*100
    print("Percentages of blue: ", round(percent,3), "%")
    print()
    cv.imshow('blue', blue)

    # green = cv.merge([z,g,z])
    green = g
    num_of_green = np.sum( green >0)
    print("num_of_blue pixels: ", num_of_green)
    percent = (num_of_green/totalpixels)*100
    print("num_of_blue pixels: ", round(percent,3), "%")
    print()
    cv.imshow('green', green)

    # red = cv.merge([z,z,r])
    red = r
    num_of_red = np.sum( red >0)
    print("num_of_blue pixels: ", num_of_red)
    percent = (num_of_red/totalpixels)*100
    print("num_of_blue pixels: ", round(percent,3), "%")
    print()
    cv.imshow('red', red)


    

    plt.show()

    cv.waitKey(0)
    break
    cv.destroyAllWindows() 