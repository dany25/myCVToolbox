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


# Load an image
## with matplotlib 
img = mpimg.imread("malik.png")
## with openCV 
imgCV = cv2.imread("malik.png")




# size of the image
h, w, ch = img.shape
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
noise =  (sigma*np.random.randn(h,w,ch)+ mu)/255.
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






















