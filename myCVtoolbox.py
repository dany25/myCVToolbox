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



#Gaussian Filtering
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





#Normalized cross-correlation for template matiching
template = img[220:251,190:221]
plt.title("template")
plt.imshow(template)
plt.show()
correlation_map = match_template(img, template)
plt.imshow(correlation_map[:,:,0], cmap="gray")
print("The template position is at: ",np.unravel_index(correlation_map.argmax()\
                        , correlation_map.shape))



# sobel gradient - magintude and orientation
circle_img = mpimg.imread("circle.png")
#circle_img= cv2.cvtColor(circle_img, cv2.COLOR_RGB2GRAY) #if
gx = sobel(circle_img,1) #WARNING: x is the column direction
gy = sobel(circle_img,0) #WARNING: y is the column direction
plt.title("x derivative") 
plt.imshow(gx,cmap="gray")
plt.show()

plt.title("y derivative") 
plt.imshow(gy,cmap="gray")
plt.show()

#we have to rescale it because the SOBEL filter of scipy is not normalized
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
#    |\) <- this is the angle
#    | \
#    |
#   \/y
    
direction = computeDirection(gx,gy)
plt.title("Sobel gradient direction")
plt.imshow(direction, cmap="gray")
plt.show()




