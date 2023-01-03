from hashlib import new
import cv2
import imutils
from matplotlib.transforms import Bbox
import numpy as np
from pytools import P
import dev_Tools as dt
import random as rd
import json 




boilerplate = '''
{
    "info": {
        "year": "2022",
        "version": "3",
        "description": "Exported from roboflow.ai",
        "contributor": "",
        "url": "https://public.roboflow.ai/object-detection/undefined",
        "date_created": "2022-07-02T07:33:39+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "clean",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "broken",
            "supercategory": "grains"
        },

        {
            "id": 2,
            "name": "dameged",
            "supercategory": "grains"
        },
        
        {
            "id": 3,
            "name": "FM",
            "supercategory": "FM"
        }

    ],
    "images": [
    ],
        "annotations":[

       
    ]
    
}
'''
json_f = json.loads(boilerplate)


picNo = 0
while True:



    number_of_object = 1500
    no_try = 100 # number of try before giving up to fit object 


    bg = cv2.imread(r'D:\datasets\Image_pool_synthetic_dataset\background_image_pool\bg (3).jpg')
    img_list = dt.allimg(r'D:\datasets\Image_pool_synthetic_dataset\starter')
    # img_list = dt.hsv_finder(r'\rice (1).jpg')

    print(bg.shape)

    blank = np.zeros(bg.shape, dtype='uint8') 
    blank1 = np.zeros(bg.shape, dtype='uint8') 
    






    i = 0
    no = 0

    list_count = []

    while i < number_of_object and no < no_try:

        ind = rd.randint(0, len(img_list)-1)

        gfo = img_list[ind]
        gfo = cv2.resize(gfo, (int(gfo.shape[1]/5), int(gfo.shape[0]/5)))


        hsv = cv2.cvtColor(gfo, cv2.COLOR_BGR2HSV)

        lower_hsv1 = np.array([0, 0, 0])
        higher_hsv1 = np.array([300, 140, 300])
        mask1 = cv2.inRange(hsv, lower_hsv1, higher_hsv1)
        gfo = cv2.bitwise_and(gfo,gfo,mask=mask1)
        rotation = rd.randint(-350,350)

        # gfo = dt.resizeimg(gfo, 500)
        # cv2.imshow('gfo', gfo)
        # cv2.waitKey(0)

        blank_inter = np.zeros(bg.shape, dtype='uint8') 

        gf = gfo
        # gf = imutils.rotate_bound(gfo, rotation)




        x_offset = rd.randint(0,bg.shape[0]-gf.shape[0])
        y_offset = rd.randint(0,bg.shape[1]-gf.shape[1])



        area_of_object = np.count_nonzero(gf)

        non_zero = np.count_nonzero(blank1[x_offset:x_offset+gf.shape[0], y_offset:y_offset+gf.shape[1]])

        iou = non_zero/(gf.shape[0]*gf.shape[1])

        # print(iou)

        # print(iou)
        # print(iou)
        # print(iou)

        if iou < 0.7 and area_of_object > 100:
            hsv = cv2.cvtColor(gf, cv2.COLOR_BGR2HSV)


            lower_hsv1 = np.array([0, 0, 12])
            higher_hsv1 = np.array([300, 300, 300])

            mask = cv2.inRange(hsv, lower_hsv1, higher_hsv1)

            # cv2.imshow('mask', mask)

            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # print(mask.shape)
            gfo = cv2.bitwise_and(gfo,gfo,mask=mask1)
            # cv2.imshow('gfo', gfo)
            # cv2.waitKey(0)
            i += 1
            # cv2.waitKey(0)


            blank_inter[x_offset:x_offset+gf.shape[0], y_offset:y_offset+gf.shape[1]] = mask + blank_inter[x_offset:x_offset+gf.shape[0], y_offset:y_offset+gf.shape[1]] 

            count_img = cv2.cvtColor(blank_inter, cv2.COLOR_BGR2GRAY)
            count_img = cv2.dilate(count_img, None, iterations=4)
            count_img = cv2.erode(count_img, None, iterations=2)

            contours, hierarchy = cv2.findContours(count_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # count = np.sort(contours)
            count = sorted(contours, key=cv2.contourArea, reverse=True)
    
            # Bbox = cv2.boxPoints(contours)

            # cv2.drawContours(blank, count[-1], -1, (0,0,255), -1)
            list_count.append(len(count[-1]))

            x,y,w,h = cv2.boundingRect(count[-1])

            # cv2.rectangle(blank, (x,y), (x+w,y+h), (0,0,255), 2)


            # bgs = dt.resizeimg(blank, 500 )           
            # cv2.imshow('blank_inter', bgs)
            # cv2.waitKey(0) 

            blank_inter2 =  cv2.bitwise_not(blank_inter)       

            # blank_inter2 = dt.resizeimg(blank_inter2, 500)
            # cv2.imshow('blank_inter2', blank_inter2)

            # cv2.waitKey(0)

            # blank_inter2 = cv2.erode(blank_inter2, None, iterations=4)

            blank = cv2.bitwise_and(blank, blank_inter2)



            blank[x_offset:x_offset+gf.shape[0], y_offset:y_offset+gf.shape[1]] = gf + blank[x_offset:x_offset+gf.shape[0], y_offset:y_offset+gf.shape[1]] 
            blank1[x_offset:x_offset+gf.shape[0], y_offset:y_offset+gf.shape[1]] = mask + blank1[x_offset:x_offset+gf.shape[0], y_offset:y_offset+gf.shape[1]] 

            # cv2.imshow('blank', blank_inter)
            annotation = {}
            annotation['id'] = i
            annotation['image_id'] = 0
            annotation['category_id'] = rd.randint(0,3) #! will be based on folder it is comming from 
            annotation['bbox'] = [x,y,w,h]
            annotation['area'] = area_of_object
            annotation['segmentation'] = np.vstack(count[-1]).squeeze().ravel().tolist()
            # print(annotation['segmentation'])
            annotation['iscrowd'] = 0
            json_f['annotations'].append(annotation)



            no = 0
        else:
            no += 1

        # break       
        # cv2.waitKey(10)

    print('object added: ', i)

    image_ano = {}
    image_ano['file_name'] = 'image_' + str(picNo) + '.jpg'
    image_ano['id'] = picNo
    image_ano['width'] = blank.shape[1]
    image_ano['height'] = blank.shape[0]
    image_ano['date_captured'] = '2019-01-01'


    {
          "id": 0,
          "license": 1,
          "file_name": "pic_test.jpg",
          "height": 3509,
          "width": 2549,
          "date_captured": "2022-07-02T07:33:39+00:00"
        }



    # x_offset = 25
    # y_offset = 532
    # blank[y_offset:y_offset+gf.shape[0], x_offset:x_offset+gf.shape[1]] = gf

    # x_offset = 50
    # y_offset = 50
    # blank[y_offset:y_offset+gf.shape[0], x_offset:x_offset+gf.shape[1]] = gf

    bg[blank != 0] = 0

    new_pic = bg + blank




    cv2.imwrite('pic_test.jpg', new_pic)
    # cv2.drawContours(new_pic, list_count, -1, (0,0,255), -1)

    blank = dt.resizeimg(blank,500)
    new_pic = dt.resizeimg(new_pic,500)
    bg = dt.resizeimg(bg,500)
    rot = imutils.rotate_bound(new_pic, -30)
    # cv2.imshow('rot', rot)

    # new = dt.resizeimg(new,500)
    cv2.imshow('blank', blank)
    cv2.imshow('dasf', new_pic)
    # cv2.imshow('bg', bg)
    blank1 = dt.resizeimg(blank1,500)
    cv2.imshow('blank1', blank1)
    cv2.waitKey(0)






with open(r'D:\agnext\Agnext\OpenCv\Synthetic datacreation\annotation_with_segmentation.json', 'w') as file:
    json.dump(json_f,file ,indent=2)