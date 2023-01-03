import cv2 as cv
from matplotlib import image
import numpy as np
import os




# resizing images but also saving it ratio
def resizeimg(img, size, interpolation = cv.INTER_CUBIC):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv.resize(mask, (size, size), interpolation)


#open all the images in a folder
def allimg(folder):
    image =load_images_from_folder(folder)
    imlist=[]
    i = 0
    for img in image:
        name = "img" + str(i)
        imlist.append(resizeimg(img, 500))
    
    return imlist
    


# load_images_from_folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


