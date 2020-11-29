import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
from PIL import Image

#Avoid using loops in Python as much as possible, especially double/triple loops etc. They are inherently slow.
#Vectorize the algorithm/code to the maximum extent possible, because Numpy and OpenCV are optimized for vector operations.
#Exploit the cache coherence.
#Never make copies of an array unless it is necessary. Try to use views instead. Array copying is a costly operation.
def pastelFilter(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #hsv[:,:,1]*= 0.25
    hsv[:,:,2]*= 2
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

def dotFilter(img):
    pass

def pilToCV(img):
    rgb = img.convert('RGB')
    cvIm = np.array(img)
    bgr = cvIm[:,:,::-1]
    return bgr

def cvToPIL(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img
        
# currently, we take in opencv images
#(x0,y0) is the top left corner of the mask
# in relation to img
# assume mask is 0
def overlayMask(img,mask,x0,y0,scale=1):
    if(x0+mask.shape[1] <= img.shape[1] and y0+mask.shape[0] <= img.shape[0]):
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                rgbTotal = sum(mask[row][col])
                if(rgbTotal < 255*3-20):
                    img[y0+row][x0+col] = mask[row][col]
    else:
        print(f'failed: {x0+mask.shape[1]}>{img.shape[1]} or {y0+mask.shape[0]} > {img.shape[0]}')
    return img

# cartoonify() inspired by the tutorial
# https://www.askaswiss.com/2016/01/how-to-create-cartoon-effect-opencv-python.html
def cartoonify(img):
    imgCartoon = img.copy()
    imgColor = imgCartoon.copy()

    pyrLevels = 3
    for i in range(pyrLevels):
        imgColor = cv2.pyrDown(imgColor)
    imgColor = cv2.bilateralFilter(imgColor,d=13,sigmaColor=15,sigmaSpace=15)
    for i in range(pyrLevels):
        imgColor = cv2.pyrUp(imgColor)
    
    imgEdge = cv2.cvtColor(imgCartoon,cv2.COLOR_BGR2GRAY)
    imgEdge = cv2.medianBlur(imgEdge,7)
    imgEdge = cv2.adaptiveThreshold(imgEdge,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=9,C=2.5)
    imgEdge = cv2.erode(imgEdge,(5,5),iterations=3)
    imgCartoon = cv2.bitwise_and(imgColor,imgColor,mask=imgEdge)
    return imgCartoon