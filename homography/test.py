import cv2 as cv
import _dev_Tools as dt
import numpy as np
 
from skimage.metrics import structural_similarity



def destortion(x,y, angle):
    x = x+(x-2040) * np.cos(angle)
    y = y+(y-1500)* np.sin(angle)
    return np.abs(x),np.abs(y)



def structural_similaritye(img1, img2, id, object1):

    if score > 0.7:
        cv.putText(img, '{}'.format(id), (x + w // 2, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 150, 0), 6)
        r = (x^2 + y^2)**0.5
        # object1.pop(id)
        print(score,"  " ,id)
        object1[id] = img[y:y+h, x:x+w,:]
        return id 

    else:
        hsv = cv.cvtColor(img[y:y+h, x:x+w,:], cv.COLOR_BGR2HSV)
        lower_hsv1 = np.array([75, 17, 0])
        higher_hsv1 = np.array([106, 225, 255])
        mask1 = cv.inRange(hsv, lower_hsv1, higher_hsv1)
        count = np.count_nonzero(mask1)/np.size(mask1)
        print(count, " ==  ", id)
        if count > 0.5:
            ook = id

        else:
            ook = None

    return ook 







object = []
rotation = [45,45, 90, 90 , 90, 90]

for i in range(1,6):
    path = r'D:\agnext\Agnext\OpenCv\homography\picture\img ('+str(i)+').jpg'
    img = cv.imread(path)

    # dt.hsv_finder(r'D:\agnext\Agnext\OpenCv\homography\picture\img (1).jpg')

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_hsv1 = np.array([80, 47, 59])
    higher_hsv1 = np.array([102, 167, 247])
    mask1 = cv.inRange(hsv, lower_hsv1, higher_hsv1)
    mask1 = cv.dilate(mask1, None, iterations=4)
    mask1 = cv.erode(mask1, None, iterations=4)# co_mask = cv.cvtColor(mask1, cv.COLOR_BINA)

    mask1 = cv.bitwise_not(mask1)
    count, her = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(len(count))

    sorted_contours = sorted(count, key=cv.contourArea, reverse=True)[:3]
    # cv.drawContours(img, sorted_contours[:3], -1, (0, 255, 0), 20)

    j = 0

    p = 0
    for c in sorted_contours:
        # cv.drawContours(img, [c], -1, (0, 255, 0), 20)
        x, y, w, h = cv.boundingRect(c)
        # print(x, y, w, h)

        cv.circle(img, (x + w // 2, y + h // 2), 20, (0, 0, 255), -1)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 225), 5)


        if i==1:
            cv.putText(img, '{}'.format(j), (x + w // 2, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 150, 0), 6)
            r = (x^2 + y^2)**0.5
            object.append(img[y:y+h, x:x+w,:])
        
        else:
            # break
            object1 = object.copy()
            for (id,point) in enumerate(object1):
                point = cv.resize(point, (w, h))
                score, grad = structural_similarity(img[y:y+h, x:x+w,:], point, multichannel=True, gradient=True)

                id = structural_similaritye(img[y:y+h, x:x+w,:], point, id, object1)
                if id == None:
                    pass
                else:
                    cv.putText(img, '{}'.format(id), (x + w // 2, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 150, 0), 6)
                    break 
                
                if p == 2:
                    cv.putText(img, '{}'.format(p), (x + w // 2, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 150, 0), 6)

                p += 1


                 

                    
            print('\n\n')
             
            

        j += 1

        # print(x, y, w, h)
    # break

    # sh_mask = dt.resizeimg(mask1, 700)
    sh_img = dt.resizeimg(img, 700)
    # cv.imshow('mask1', sh_mask)
    cv.imshow('img', sh_img)
 
    cv.waitKey(0)

# for ing in object:
#     cv.imshow('img', ing)
#     cv.waitKey(0)
