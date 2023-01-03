# import cv2 as cv
# from matplotlib import image
# import numpy as np
# import matplotlib.pyplot as plt
import dev_Tools as dt 
# from skimage.filters import unsharp_mask
# from skimage import img_as_ubyte, io
# import matplotlib.pyplot as plt
# import numpy as np
# import pyclesperanto_prototype as cle

# img = cv.imread(r"D:\datasets\scanner bin\contaminated_grainframe79.png")
# input_image_original = img_as_ubyte(io.imread(r"D:\datasets\scanner bin\contaminated_grainframe79.png", as_gray=True))
# input_image = np.invert(input_image_original)
# binary = cle.binary_not(cle.threshold_otsu(input_image))
# labels = cle.voronoi_labeling(binary)

# # cv.imshow(img)
# cle.imshow(labels, labels=True)

# cv.waitKey(2200)




# # image_folder_path = r'D:\datasets\scanner bin'

# # img_list = dt.allimg(image_folder_path)[23:]

# # i = 0
# # for img in img_list:
# #     fig, axes = plt.subplots(nrows=2, ncols=3,
# #                          sharex=True, sharey=True, figsize=(10, 10))

# #     ax = axes.ravel()
    
# #     result_1 = unsharp_mask(img, radius=5, amount=2)

# #     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# #     result_2 = cv.Canny(img, 100, 200)

# #     # result_1a   = cv.cvtColor(result_1, cv.COLOR_BGR2GRAY) 
# #     # result_3 = cv.Canny(result_1a, 100, 200)

# #     print(i)
# #     i += 1

# #     image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# #     ax[0].imshow(image, cmap=plt.cm.gray)
# #     # ax[0].set_title('Original image')
# #     ax[1].imshow(result_1, cmap=plt.cm.gray)
# #     # ax[1].set_title('Enhanced image, radius=1, amount=1.0')
# #     ax[2].imshow(result_2, cmap=plt.cm.gray)
# #     # ax[2].set_title('Enhanced image, radius=5, amount=2.0')
# #     ax[3].imshow(img, cmap=plt.cm.gray)
# #     # ax[2].set_title('Enhanced image, radius=5, amount=2.0')

    
# #     for a in ax:
# #         a.axis('off')
# #     fig.tight_layout()
# #     plt.show()

# #     break
# #     if cv.waitKey(1) & 0xFF == ord('q'):
# #         break



#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=g3OZJ6skE_U


"""
This code performs grain size distribution analysis and dumps results into a csv file.
Step 1: Read image and define pixel size (if needed to convert results into microns, not pixels)
Step 2: Denoising, if required and threshold image to separate grains from boundaries.
Step 3: Clean up image, if needed (erode, etc.) and create a mask for grains
Step 4: Label grains in the masked image
Step 5: Measure the properties of each grain (object)
Step 6: Output results into a csv file
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io

#STEP1 - Read image and define pixel size
image_path = r'D:\datasets\scanner bin\contaminated_grainframe79.png'
img = cv2.imread(r"D:\datasets\scanner bin\contaminated_grainframe79.png", 0)
# dt.hsv_finder(r"D:\datasets\scanner bin\contaminated_grainframe79.png")

pixels_to_um = 0.5 # (1 px = 500 nm)

#cropped_img = img[0:450, :]   #Crop the scalebar region

#Step 2: Denoising, if required and threshold image

#No need for any denoising or smoothing as the image looks good.
#Otherwise, try Median or NLM
#plt.hist(img.flat, bins=100, range=(0,255))

#Change the grey image to binary by thresholding. 

img = dt.background_remover(image_path, lowH = 89, highH = 124, lowS = 149, highS = 255, lowV = 147, highV = 255)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imshow('thresh', thresh)

#print(ret)  #Gives 157 on grains2.jpg. OTSU determined this to be the best threshold. 

#View the thresh image. Some boundaries are ambiguous / faint.
#Some pixles in the middle. 
#Need to perform morphological operations to enhance.

#Step 3: Clean up image, if needed (erode, etc.) and create a mask for grains

kernel = np.ones((3,3),np.uint8) 
eroded = cv2.erode(thresh,kernel,iterations = 1)
dilated = cv2.dilate(eroded,kernel,iterations = 1)

# Now, we need to apply threshold, meaning convert uint8 image to boolean.
mask = dilated == 255  #Sets TRUE for all 255 valued pixels and FALSE for 0
#print(mask)   #Just to confirm the image is not inverted. 

#from skimage.segmentation import clear_border
#mask = clear_border(mask)   #Removes edge touching grains. 

io.imshow(mask)  #cv2.imshow() not working on boolean arrays so using io
#io.imshow(mask[250:280, 250:280])   #Zoom in to see pixelated binary image

#Step 4: Label grains in the masked image

#Now we have well separated grains and background. Each grain is like an object.
#The scipy ndimage package has a function 'label' that will number each object with a unique ID.

#The 'structure' parameter defines the connectivity for the labeling. 
#This specifies when to consider a pixel to be connected to another nearby pixel, 
#i.e. to be part of the same object.

#use 8-connectivity, diagonal pixels will be included as part of a structure
#this is ImageJ default but we have to specify this for Python, or 4-connectivity will be used
# 4 connectivity would be [[0,1,0],[1,1,1],[0,1,0]]
s = [[1,1,1],[1,1,1],[1,1,1]]
#label_im, nb_labels = ndimage.label(mask)
labeled_mask, num_labels = ndimage.label(mask, structure=s)

#The function outputs a new image that contains a different integer label 
#for each object, and also the number of objects found.


#Let's color the labels to see the effect
img2 = color.label2rgb(labeled_mask, bg_label=0)

cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)

#View just by making mask=threshold and also mask = dilation (after morph operations)
#Some grains are well separated after morph operations

#Now each object had a unique number in the image. 
#Total number of labels found are...
#print(num_labels) 

#Step 5: Measure the properties of each grain (object)

# regionprops function in skimage measure module calculates useful parameters for each object.

clusters = measure.regionprops(labeled_mask, img)  #send in original image for Intensity measurements

#The output of the function is a list of object properties. 

#Test a few measurements
#print(clusters[0].perimeter)

#Can print various parameters for all objects
#for prop in clusters:
#    print('Label: {} Area: {}'.format(prop.label, prop.area))
    
#Step 6: Output results into a csv file   
#Best way is to output all properties to a csv file
    
propList = ['Area',
            'equivalent_diameter', #Added... verify if it works
            'orientation', #Added, verify if it works. Angle btwn x-axis and major axis.
            'MajorAxisLength',
            'MinorAxisLength',
            'Perimeter',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity']    
    

output_file = open('image_measurements.csv', 'w')
output_file.write(',' + ",".join(propList) + '\n') #join strings in array by commas, leave first cell blank
#First cell blank to leave room for header (column names)

for cluster_props in clusters:
    #output cluster properties to the excel file
    output_file.write(str(cluster_props['Label']))
    for i,prop in enumerate(propList):
        if(prop == 'Area'): 
            to_print = cluster_props[prop]*pixels_to_um**2   #Convert pixel square to um square
        elif(prop == 'orientation'): 
            to_print = cluster_props[prop]*57.2958  #Convert to degrees from radians
        elif(prop.find('Intensity') < 0):          # Any prop without Intensity in its name
            to_print = cluster_props[prop]*pixels_to_um
        else: 
            to_print = cluster_props[prop]     #Reamining props, basically the ones with Intensity in its name
        output_file.write(',' + str(to_print))
    output_file.write('\n')
output_file.close()   #Closes the file, otherwise it would be read only. 
