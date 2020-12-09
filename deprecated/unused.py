import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
from PIL import Image
from cvhelpers import *


def pastelFilter(img):
    bgr = img
    hsv = cvtHSV(img)
    h,s,v = np.squeeze(np.split(hsv,3,axis=2))
    
    rMask = np.where(h<=30,h,0)
    rMask = np.where(h>30,rMask,1)
    yMask = np.where(30 < h.all() and h.all() <=60,h,0)
    yMask = np.where(h.all()>60 or h.all()<=30,yMask,1)
    gMask = np.where(60<h.all()<=90,h,0)
    gMask = np.where(h.all()>90 or h.all()<=60,gMask,1)
    cMask = np.where(90<h.all()<=120,h,0)
    cMask = np.where(h.all()>120 or h.all()<=90,cMask,1)
    bMask = np.where(120<h.all()<=150,h,0)
    bMask = np.where(h.all()>150 or h.all()<=120,bMask,1)
    mMask = np.where(150<h.all()<=180,h,0)
    mMask = np.where(h.all()>180 or h.all()<=150,mMask,1)
    

    rColored = rMask*30#hsv[:,:,1]*.1
    yColored = yMask*60#hsv[:,:,1]*.2
    gColored = gMask*90#hsv[:,:,1]*.3
    cColored = cMask*120#hsv[:,:,1]*.4
    bColored = bMask*150#hsv[:,:,1]*.5
    mColored = mMask*180#hsv[:,:,1]*.6

    hsv[:,:,0] = rColored+yColored+gColored+cColored+bColored+mColored
    #hsv[:,:,1] = hsv[:,:,1]*0# s lower
    hsv[:,:,2] = hsv[:,:,2]*2 # v higher
    
    #hsv[:,:,1] = hsv[:,:,1] * 0.5
    #hsv[:,:,2] = hsv[:,:,2] / ((hsv[:,:,2])/256)**2
    hsv = hsv.astype('uint8')
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return rgb

def convSobel(n):
    #smooth = np.array(
    #    [[ 1, 2, 1 ]], dtype='int'
    #)
    #smootht = np.array([[1],[2],[1]],dtype='int')
    #kernel = smootht*smooth*1/8 # kernel to transform into bigger kernel
    #sob3 = smootht*np.array([1,0,-1],dtype='int') # starting 3x3
    sobelX = np.array(( [-1,0,1],[-2,0,2],[-1,0,1] ))
    sobelY = np.array(( [-1,-2,-1],[0,0,0],[1,2,1] ))
    kernel = sobelX
    u1, v1, sig = svdKern(kernel)
    inW,inH,kW,kH,padW,padH,image = helper(inp,kernel)

    final = sob3
    passes = n//2 -1 #3x3 = 0 passes
    
    for i in range(passes):
        inW,inH = final.shape[1],final.shape[0]
        
        output = np.empty(final.shape,dtype='float32')
        
        padW,padH = kW//2,kH//2
        image = np.ones((inH+2*padH,inW+2*padW),dtype='float32')
        image[padH:inH+padH,padW:inW+padW] = final
        
        for x in np.arange(0,inW):
            for y in np.arange(0,inH):
                roi = (image[y:y+2*padH+1,x:x+2*padW+1]*u1*v1*s).sum() # a 3d array
                output[y,x] = roi # 3 elem array?
        arrMax = np.amax(output)
        arrMin = np.amin(output)
        factor = (arrMax-arrMin+1)/256
        output[:,:] = (output[:,:]-arrMin)//factor
        output = output.astype("uint8")
        final = output * 1/16
    return final

# cartoonify() inspired by the tutorial
# https://www.askaswiss.com/2016/01/how-to-create-cartoon-effect-opencv-python.html
def cartoonify(img):
    #imgColor = img.copy()
    imgColor = img.copy() # avoiding copying?

    pyrLevels = 3
    biPasses = 7
    for i in range(pyrLevels):
        imgColor = cv2.pyrDown(imgColor)
    for i in range(biPasses):
        imgColor = cv2.bilateralFilter(imgColor,d=9,sigmaColor=9,sigmaSpace=7)
    #imgColor = cv2.bilateralFilter(imgColor,d=13,sigmaColor=15,sigmaSpace=15)
    for i in range(pyrLevels-1):
        imgColor = cv2.pyrUp(imgColor)
    
    imgColor = cv2.pyrUp(imgColor,(img.shape))
    imgEdge = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgEdge = cv2.medianBlur(imgEdge,7)
    imgEdge = cv2.adaptiveThreshold(imgEdge,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=9,C=2.5)
    imgEdge = cv2.erode(imgEdge,(5,5),iterations=3)
    img = cv2.bitwise_and(imgColor,imgColor,mask=imgEdge)
    return img

# standard convolution kernels from https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
def testConvolve(img):
    # construct average blurring kernels used to smooth an image
    smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (15 * 15))
    gauss = np.array((
        [1,4,6,4,1],
        [4,16,24,16,4],
        [6,24,36,24,6],
        [4,16,24,16,4],
        [1,4,6,4,1]
    )) * (1.0 / (16*16))
    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")
    # construct the Sobel x-axis kernel
    sobelX = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")
    # construct the Sobel y-axis kernel
    sobelY = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int")
    sobel = np.array((
        [1,  2, 0,  -2, -1],
        [4, 8, 0,  -8, -4],
        [6, 12, 0, -12, -6],
        [4,  8, 0,  -8, -4],
        [1,  2, 0,  -2, -1]), dtype = 'int')
    
    #convolve = Convolution(nc_in, nc_out, kernel_size, stride=2,padding=1)
    return pastelFilter(img)

