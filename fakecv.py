import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
from PIL import Image


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

    newImg = copy.deepcopy(img)
    #print(f"img: {newImg.shape}, mask: {mask.shape}, xy: {x0,y0}")
    if(x0+mask.shape[1] <= img.shape[1] and y0+mask.shape[0] <= img.shape[0]):
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                rgbTotal = sum(mask[row][col])
                if(rgbTotal < 255*3-20):
                    newImg[y0+row][x0+col] = mask[row][col]
    else:
        print(f'failed: {x0+mask.shape[1]}>{img.shape[1]} or {y0+mask.shape[0]} > {img.shape[0]}')
    return newImg