#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:22:08 2018

@author: daniel
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import sobel
from skimage.feature import match_template
import math

# Load an image
## with matplotlib 
img = mpimg.imread("malik.png")
## with openCV 
imgCV = cv2.imread("malik.png")




# size of the image
height, width, channels = img.shape
print("image size: ",img.shape) # (height, width)




# Display an image
## with matplotlib 
print("display with matplotlib")
plt.imshow(img)
plt.axis("off") # to hide tick values on X and Y axis
plt.show()

## with openCV
print("display with openCV")
plt.axis("off")
plt.imshow(cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGB))
plt.show()




# Add gaussian noise
print("############# Adding Gaussian noise ############# ")
plt.title("one line on the image")
plt.plot(img[200,:,0])
plt.show()

mu, sigma= 0, 3.
noise =  (sigma*np.random.randn(height,width,channels)+ mu)/255.
noisy_image = img +noise

plt.title("same line on the noisy image")
plt.plot(noisy_image[200,:,0])
plt.show()



# Make the difference of two images the proper way
img2 = mpimg.imread("papa.png")
img2 = img2[:,:,:3]
print("img2 size: ",img2.shape)
plt.imshow(img2)
plt.axis("off")
plt.show()
#TODO
img_diff = (img - img2) + (img2 - img)
plt.imshow(img_diff)
plt.show()




# Remaping of the range of values in an image
"""
black_image = np.zeros((600,400))
plt.imshow(black_image,cmap='gray')
plt.plot()
plt.imshow(black_image+noise[:,:,0],cmap='gray')
plt.plot()
"""



# Gaussian Filtering
print("############# Gaussian Filtering ############# ")
## manual
def fgaussian(size, sigma):
     m,n = size,size
     h, k = m//2, n//2
     x, y = np.mgrid[-h:h, -k:k]
     return np.exp(-(x**2 + y**2)/(2*sigma**2))
h= fgaussian(31,10)
plt.title("Gaussian Kernel")
plt.imshow(h)
plt.show()
## with scipy 
     #by default we deal with the boundary pixels with the 'reflect' mode
img_blurred = gaussian_filter(img, sigma=7)
plt.title("img blurred")
plt.axis("off")
plt.imshow(img_blurred)
plt.show()





# Normalized cross-correlation for template matiching
print("############# Template matching ############# ")
template = img[220:251,190:221]
plt.title("template")
plt.imshow(template)
plt.show()
correlation_map = match_template(img, template)
plt.imshow(correlation_map[:,:,0], cmap="gray")
print("The template position is at: ",np.unravel_index(correlation_map.argmax()\
                        , correlation_map.shape))



# Kernel: 2D Gaussian
print("############# Kernels and its derivative ############# ")
## The convolution with a filter and an impulse is the filter
def gaussian_kernel(size, sigma):
    gKernel = np.zeros((size,size))
    gKernel[int(size/2),int(size/2)]=1
    return gaussian_filter(gKernel,sigma)
plt.imshow(gaussian_kernel(31,7))

# Kernel: Derivative of Gaussian filter (with sobel operator)
xDerivativeKernel = sobel(gaussian_kernel(31,7),1)
plt.title("x-derivative kernel")
plt.imshow(xDerivativeKernel)
plt.show()
yDerivativeKernel = sobel(gaussian_kernel(31,7),0)
plt.title("y-derivative kernel")
plt.imshow(yDerivativeKernel)
plt.show()



# Kernel: Lplacian of gaussian filter
# TODO



# sobel gradient - magintude and orientation
print("############# Sobel Gradient ############# ")
# we are doing an edge detection with the sobel operator
circle_img = mpimg.imread("circle.png")
#circle_img= cv2.cvtColor(circle_img, cv2.COLOR_RGB2GRAY) 
gx = sobel(circle_img,1) #WARNING: x is the column direction
gy = sobel(circle_img,0) #WARNING: y is the column direction
plt.title("x derivative") 
plt.imshow(gx,cmap="gray")
plt.show()

plt.title("y derivative") 
plt.imshow(gy,cmap="gray")
plt.show()

# we have to rescale it because the SOBEL filter of scipy is not normalized
# We can see this by looking at the response of the correlation of the sobel 
# x filter on the image [0|1] meaning one pixel black, one pixel white. We have
# to obtain [0|1] after the convolution
magnitude = np.sqrt((gx)**2+(gy)**2)/(4*np.sqrt(2))
plt.title("Sobel gradient magnitude")
#plt.imshow(np.sqrt((gx-0.5)**2+(gy-0.5)**2), cmap="gray")
plt.imshow(magnitude, cmap="gray")
plt.show()

def angleInDegree(dx,dy):
    if (not dx):
        return 0
    return math.atan(dy/dx)*180./math.pi

def computeDirection(gx,gy):
    n,m = gx.shape[:2]
    res = np.zeros((gx.shape))
    for i in range(n):
        for j in range(m):
            res[i,j]=angleInDegree(gx[i,j],gy[i,j])
    return res
# the direction lies between -Pi/2 and Pi/2 But be careful the angle is taken 
#   with the convention that positive y goes down. 
#    _____>x
#    |\)theta <- this is the angle
#    | \
#    |
#   \/y
direction = computeDirection(gx,gy)
plt.title("Sobel gradient direction")
plt.imshow(direction, cmap="gray")
plt.show()



# Canny edge detector:
print("############# Canny Edge detector ############# ")
grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
plt.imshow(grayImg,cmap="gray")
plt.show()
cannyEdges = cv2.Canny(np.uint8(grayImg*255),20,200) #accept only uint8 images
plt.title("Canny edge detector")
plt.imshow(cannyEdges,cmap="gray")
plt.show()



# Hough transform : line detection
print("############# Hough transform : line detection ############# ")
# TODO







