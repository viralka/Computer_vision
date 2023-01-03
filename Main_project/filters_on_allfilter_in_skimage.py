from calendar import c
from email.errors import HeaderMissingRequiredValue
from statistics import median
import cv2 as cv
from matplotlib.cbook import file_requires_unicode 
import numpy as np
from scipy.fftpack import diff
import image_opener as ip
import skimage.filters as sf
import skimage.morphology as sm
from skimage import data
from skimage.filters import unsharp_mask
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.data import astronaut
from skimage.filters import window

imgo = cv.imread(r"D:\datasets\scanner bin\contaminated_grainframe79.png")
# img = ip.resizeimg(imgo, 700)
img = imgo
imggr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("window",img)

# filering  
#! error
# sf.apply_hysteresis_threshold(imggr, 0.5, .4)
# cv.imshow("hysteresis",imggr)

# # butterworth
# high_pass = sf.butterworth(img, 0.07, True, 8)
# low_pass = sf.butterworth(img, 0.01, False, 4, channel_axis=-1)
# cv.imshow("butterworth_high",high_pass)
# cv.imshow("butterworth_low",low_pass)
# cv.waitKey(2200)
# cv.destroyAllWindows()

# # correlate_sparse


result_8 = cv.Canny(imggr, 100, 200)

# # difference_of_gaussians used in edge and blob detection

filterd_image = sf.difference_of_gaussians(img, 1, 50, channel_axis=-1)
# filtered_image = sf.difference_of_gaussians(imggr, (2,5), (3,20))
# cv.imshow("filterd image",filterd_image)
result_7 = filterd_image
cv.waitKey(2200)
cv.destroyAllWindows()

# # fardi  finding edge magnitude  # gray scale image 
# fardi = sf.farid(imggr)
# cv.imshow("fardi",fardi)
# cv.waitKey(2200)
# cv.destroyAllWindows()


# # fardi_h for horizantal  edge detection 
# fardi = sf.farid_h(imggr)
# cv.imshow("fardi",fardi)
# cv.waitKey(2200)
# cv.destroyAllWindows()

# # fardi_v for horizantal  edge detection 
# fardi = sf.farid_v(imggr)
# cv.imshow("fardi",fardi)
# cv.waitKey(2200)
# cv.destroyAllWindows()

# # fardi_ used to detect tube like structuer in image

# frangi = sf.frangi(imggr, sigmas=range(1, 10, 2), alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode='reflect', cval=0)# fardi_h for horizantal  edge detection 

# cv.imshow("fardi",frangi)
# cv.waitKey(2200)
# cv.destroyAllWindows()

# # gabor_filter #? this filter can do the trick 

# filter_real , filt_img = sf.gabor(imggr, frequency=0.6)
# cv.imshow("filter_real",filter_real)

# filter_real , filt_img = sf.gabor(imggr, frequency=0.1)
# cv.imshow("filter_real",filter_real)


# # gaussian 
# gaussian = sf.gaussian(imggr, sigma=1, mode='mirror', cval=0)
# cv.imshow("gaussian",gaussian)

# # hessian  #! this motherfucker did the trick 
hessian = sf.hessian(imggr, mode='mirror', cval=0)
result_6 = hessian
# cv.imshow("hessian",hessian)

# # inverse #? not understood
# inverse = sf.inverse(imggr, impulse_response=)
# cv.imshow("inverse",inverse)

# # laplacian 
# laplacian = sf.laplace(imggr, ksize = 11, mask = None)
# cv.imshow("laplacian",laplacian)


# # median #? not understood
# median = sf.median(imggr, sm.disk(3))
# cv.imshow("median",median)

# meijering #! can be used for depth perseption 
meijering_color = sf.meijering(imggr, sigmas=range(1, 10, 2), alpha=None, black_ridges=True, mode='reflect', cval=0)
meijering_gray = sf.meijering(img, sigmas=range(1, 10, 2), alpha=None, black_ridges=True, mode='reflect', cval=0)
result_5 = meijering_gray
# cv.imshow("meijering_color",meijering_color)
# cv.imshow("meijering_gray",meijering_gray)

# # prewitt # edge detection
# prewitt = sf.prewitt(imggr, mode='mirror', cval=0)
# cv.imshow("prewitt",prewitt)

# # prewitt_h # edge detection
# prewitt_h = sf.prewitt_h(imggr, mask = None)
# cv.imshow("prewitt_h",prewitt_h)

# # prewitt_v # edge detection
# prewitt_v = sf.prewitt_v(imggr, mask = None)
# cv.imshow("prewitt_v",prewitt_v)

# # rank_order #? not understood
# label , orignal_vlue  = sf.rank_order(imggr)
# print(label[0])
# print(orignal_vlue)

# # # roberts # edge detection
# roberts = sf.roberts(imggr, mask = None)
# cv.imshow("roberts",roberts)

# # roberts_neg_diag # edge detection
# roberts_neg_diag = sf.roberts_neg_diag(imggr, mask = None)
# cv.imshow("roberts_neg_diag",roberts_neg_diag)

# # roberts_pos_diag # edge detection
# roberts_pos_diag = sf.roberts_pos_diag(imggr, mask = None)
# cv.imshow("roberts_pos_diag",roberts_pos_diag)

# sato # vessel detection #! this some thing 
sato = sf.sato(img, sigmas=range(1, 10, 2), black_ridges=True, mode='reflect', cval=0)
result_5 = sato
# cv.imshow("sato",sato)

# # scharr # edge magnitude
# scharr = sf.scharr(imggr, mode='mirror', cval=0)
# cv.imshow("scharr",scharr)

# # scharr_h # edge magnitude
# scharr_h = sf.scharr_h(imggr, mask = None)
# cv.imshow("scharr_h",scharr_h)

# # scharr_v # edge magnitude
# scharr_v = sf.scharr_v(imggr, mask = None)
# cv.imshow("scharr_v",scharr_v)

# # sobel # edge magnitude
# sobel = sf.sobel(imggr, mode='mirror', cval=0)
# cv.imshow("sobel",sobel)

# sobel_h # edge magnitude
sobel_h = sf.sobel_h(imggr, mask = None)
result_5 = sobel_h
# cv.imshow("sobel_h",sobel_h)


# # sobel_v # edge magnitude
# sobel_v = sf.sobel_v(imggr, mask = None)
# cv.imshow("sobel_v",sobel_v)

############## see threshoalding in detail ##############
# # # # threshold_isodata 
# # # threshold_isodata = sf.threshold_isodata(imggr)
# # # print(threshold_isodata)


# # # # threshold_li #? not understood
# # # threshold_li = sf.threshold_li(imggr)
# # # print(threshold_li)


# # # # threshold_local #? not understood
# # # threshold_local = sf.threshold_local(imggr, block_size=11, method='gaussian')
# # # print(threshold_local)

# # from skimage.data import camera
# # image = camera()[:50, :50]
# # binary_image1 = image > sf.threshold_local(image, 15, 'mean')
# # func = lambda arr: arr.mean()
# # binary_image2 = image > sf.threshold_local(image, 15, 'generic',param=func)
# # cv.imshow("binary_image1",binary_image1)

# kmean clustering 
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 10
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
result_4 = cv.cvtColor(res2,cv.COLOR_BGR2RGB) #? 
# cv.imshow('k_mean',res2)
# cv.imwrite('bg7.jpg',res2)




# #  unsharp_mask #! this thing can do the trick 3 one

image = imggr
result_1 = unsharp_mask(image, radius=1, amount=1)
result_2 = unsharp_mask(image, radius=5, amount=2)
result_3 = unsharp_mask(image, radius=20, amount=1)

fig, axes = plt.subplots(nrows=3, ncols=3,
                         sharex=True, sharey=True, figsize=(10, 10))

ax = axes.ravel()


ax[0].imshow(image, cmap=plt.cm.gray)
# ax[0].set_title('Original image')
ax[1].imshow(result_1, cmap=plt.cm.gray)
# ax[1].set_title('Enhanced image, radius=1, amount=1.0')
ax[2].imshow(result_2, cmap=plt.cm.gray)
# ax[2].set_title('Enhanced image, radius=5, amount=2.0')
ax[3].imshow(result_3, cmap=plt.cm.gray)
# ax[3].set_title('Enhanced image, radius=20, amount=1.0')
ax[4].imshow(result_4, cmap=plt.cm.gray)
# ax[4].set_title('Enhanced image, radius=20, amount=1.0')
ax[5].imshow(result_5, cmap=plt.cm.gray)
# ax[5].set_title('Enhanced image, radius=20, amount=1.0')
ax[6].imshow(result_6, cmap=plt.cm.gray)
# ax[6].set_title('Enhanced image, radius=20, amount=1.0')
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
ax[7].imshow(img, cmap=plt.cm.gray)
# ax[7].set_title('Enhanced image, radius=20, amount=1.0')
ax[8].imshow(result_8, cmap=plt.cm.gray)
# ax[8].set_title('Enhanced image, radius=20, amount=1.0')


for a in ax:
    a.axis('off')
fig.tight_layout()
plt.show()


# # windows #? how to use this
# wimg = imggr * sf.window("hann", imggr.shape)
# cv.imshow("wing",wimg)
# image = img_as_float(rgb2gray(img))

# wimage = image * window('hann', image.shape)

# image_f = np.abs(fftshift(fft2(image)))
# wimage_f = np.abs(fftshift(fft2(wimage)))

# fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# ax = axes.ravel()
# ax[0].set_title("Original image")
# ax[0].imshow(image, cmap='gray')
# ax[1].set_title("Windowed image")
# ax[1].imshow(wimage, cmap='gray')
# ax[2].set_title("Original FFT (frequency)")
# ax[2].imshow(np.log(image_f), cmap='magma')
# ax[3].set_title("Window + FFT (frequency)")
# ax[3].imshow(np.log(wimage_f), cmap='magma')
# plt.show()

# # LPIFilter2D #? have to see more into it 

# def filt_func(r, c, sigma = 1):
#     return np.exp(-np.hypot(r, c)/sigma)
# filter = sf.LPIFilter2D(filt_func)
# image = img_as_float(rgb2gray(img))
# filtered = filter(image)
# fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# ax = axes.ravel()
# ax[0].set_title("Original image")
# ax[0].imshow(image, cmap='gray')
# ax[1].set_title("Filtered image")
# ax[1].imshow(filtered, cmap='gray')
# plt.show()






cv.waitKey(0)