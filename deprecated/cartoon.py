import math, copy, random, time, string
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from tkinter import *
from cmu_112_graphics import *



'''
live = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = live.read()

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Display the resulting frame
    cv.imshow('frame',gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

img = cv.imread()
# downsample image using Gaussian pyramid
img_color = cv.bilateralFilter(img,50,100,100)
# convert to grayscale and apply median blur
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img_blur = cv.medianBlur(img_gray, 7)
img_edge = cv.adaptiveThreshold(img_blur, 255,
                                 cv.ADAPTIVE_THRESH_MEAN_C,
                                 cv.THRESH_BINARY,
                                 blockSize=9,
                                 C=2)

# convert back to color, bit-AND with color image
img_edge = cv.cvtColor(img_edge, cv.COLOR_GRAY2RGB)
img_cartoon = cv.bitwise_and(img_color, img_edge)

#kernel = np.ones((10,10),np.float32)/25

#dst = cv.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_cartoon),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

'''

live = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = live.read()

    img_cartoon = cv.GaussianBlur(frame, (7,7),50)
    img_cartoon = cv.Canny(img_cartoon, 50,50)


    '''
    num_down = 2       # number of downsampling steps
    num_bilateral = 7  # number of bilateral filtering steps

    img_rgb = frame

    # downsample image using Gaussian pyramid
    img_color = img_rgb
    for _ in range(num_down):
        img_color = cv.pyrDown(img_color)

    # repeatedly apply small bilateral filter instead of
    # applying one large filter
    for _ in range(num_bilateral):
        img_color = cv.bilateralFilter(img_color, d=9,
                                    sigmaColor=9,
                                    sigmaSpace=7)

    # upsample image to original size
    for _ in range(num_down):
        img_color = cv.pyrUp(img_color)

    # convert to grayscale and apply median blur
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    img_blur = cv.medianBlur(img_gray, 7)
    img_edge = cv.adaptiveThreshold(img_blur, 255,
                                    cv.ADAPTIVE_THRESH_MEAN_C,
                                    cv.THRESH_BINARY,
                                    blockSize=9,
                                    C=2)

    # convert back to color, bit-AND with color image
    img_edge = cv.cvtColor(img_edge, cv.COLOR_GRAY2RGB)
    img_cartoon = cv.bitwise_and(img_color, img_edge)
    '''
    # Display the resulting frame
    imgStack = stackImages(0.6,[[frame, img_cartoon]])
    cv.imshow('frame',imgStack)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


live.release()
cv.destroyAllWindows()
