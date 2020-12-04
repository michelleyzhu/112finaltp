import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
from PIL import Image


def cvtHSV(bgr):
    smallBgr = bgr.astype('float32')/255
    b,g,r = bgr[:,:,0],bgr[:,:,1],bgr[:,:,2]
    smallB,smallG,smallR = smallBgr[:,:,0],smallBgr[:,:,1],smallBgr[:,:,2]
    # cmax is the 2-d(flattened) array of the maximum of each element
    cmax = np.amax(smallBgr,axis=2)
    cmin = np.amin(smallBgr,axis=2)
    diff = cmax-cmin

    val = cmax*255

    bMask = 1 - np.sign(cmax-smallB) # either 0(is not the max) or 1(is the max)
    gMask = 1 - np.sign(cmax-smallG)
    rMask = 1 - np.sign(cmax-smallR)

    #cv2.imwrite('bMask.jpg',bMask*255)
    #cv2.imwrite('gMask.jpg',gMask*255)
    #cv2.imwrite('rMask.jpg',rMask*255)

    # deal with 0 division by masking 0-val diffs to arbitrary number(1)
    diffCopy = np.where(diff != 0, diff,1)

    hueB = (60 * ((smallR - smallG) / diffCopy) + 240) % 360
    hueG = (60 * ((smallB - smallR) / diffCopy) + 120) % 360
    hueR = (60 * ((smallG - smallB) / diffCopy) + 360) % 360
    
    hue = (hueB*bMask + hueG*gMask + hueR*rMask)/2
    
    cmaxCopy = np.where(cmax != 0, cmax, 1)
    sat = 255*(diff)/cmaxCopy
    
    return np.dstack([hue,sat,val])

def pilToCV(img):
    rgb = img.convert('RGB')
    cvIm = np.array(img)
    bgr = cvIm[:,:,::-1]
    return bgr

def cvToPIL(img):
    img = cv2.cvtColor(img.astype('uint8'),cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img
        
def maskToRGB(mask):
    return mask*255

def helper(inp,kernel):
    isRGB = len(inp.shape) == 3
    inp = inp.astype('float32')

    inW,inH = inp.shape[1],inp.shape[0]
    kW,kH = kernel.shape[1],kernel.shape[0]
    padW,padH = kW//2,kH//2
    if(isRGB):
        image = np.ones((inH+2*padH,inW+2*padW,inp.shape[2]),dtype='float32')
    else:
        image = np.ones((inH+2*padH,inW+2*padW),dtype='float32')
    image[padH:inH+padH,padW:inW+padW] = inp
    
    # top border
    image[:padH,padW:-padW] = inp[padH:0:-1]
    image[-padH:,padW:-padW] = inp[inH:-padH-1:-1]
    # side border
    image[padH:-padH,:padW] = inp[:,padW:0:-1]
    image[padH:-padH,-padW:] = inp[:,inW:-padW-1:-1]
    # corners
    image[:padH,    :padW]      = inp[:padH, :padW]
    image[:padH,    -padW:]     = inp[:padH, -padW:]
    image[-padH:,   :padW]      = inp[-padH:, :padW]
    image[-padH:,   -padW:]     = inp[-padH:, -padW:]
    
    return inW,inH,kW,kH,padW,padH,image

def svdKern(kernel):
    u, s, vh = np.linalg.svd(kernel)
    u1 = np.transpose(np.array([u[:,0]]))
    v1 = vh[0,:]
    sig = s[0]
    return u1,v1,sig

def scaleRGB(output):
    arrMax = np.amax(output)
    arrMin = np.amin(output)
    factor = (arrMax-arrMin+1)/256
    output = (output-arrMin)/factor
    output = output.astype("uint8")
    return output

def gaussianGen(n=5,sig=1):
    kernel = np.zeros((n,n))
    mean = n//2
    for x in range(n):
        for y in range(n):
            kernel[x,y] = np.exp( -0.5 * (((x-mean)/sig)**2 + ((y-mean)/sig)**2))/(2 * math.pi * sig**2)
    kernel = kernel/kernel.sum()
    return kernel
    
def pyrD(img):
    g5 = gaussianGen()

    # toggle these three lines to run your own version!
    blur = cv2.filter2D(img,3,g5)
    blur = scaleRGB(blur)
    #blur = genConvolve(img,g5)

    smaller = blur[::2,::2]
    return smaller

def pyrU(img,size=None):
    if(size==None):
        size = img.shape[0]*2,img.shape[1]*2,img.shape[2]
    #bigger = np.zeros(size)
    med = np.median(img)
    bigger = np.full_like(img,med,shape=size)
    bigger[::2,::2] = img
    bigger[1::2,1::2] = img
    gup = gaussianGen()

    # toggle these three lines to run your own version!
    bigger = cv2.filter2D(bigger,3,4*gup)
    bigger = scaleRGB(bigger)
    #bigger = genConvolve(bigger, 4*gup)
    
    return bigger

def cvtGray(img): # converts bgr to gray
    img = 0.114*img[:,:,0] + 0.587*img[:,:,1] + 0.299*img[:,:,2]
    return img.astype('uint8')

def adapMask(img,kernel,n,C=0):
    if(kernel == 'mean'):
        kernel = np.ones((n,n))/ (n**2)
        channels = 3
    elif(kernel == 'gauss'):
        kernel = gaussianGen(n=n)
        channels = 3
    
    thresholds = cv2.filter2D(img,channels,kernel)
    #thresholds = genConvolve(img,kernel,scale=False)
    
    diff = img.astype('float32')-thresholds+C
    mask = ((np.sign(diff)+1)/2).astype('uint8')
    above = mask*img
    return mask

def applyMask(img,mask):
    # mask is 1/0 vals
    # img is rgb
    # looking for img on highlighted 1's
    return np.dstack([img[:,:,0]*mask,img[:,:,1]*mask,img[:,:,2]*mask])

def medBlur(inp,n):
    kernel = np.ones((n,n))
    inW,inH,kW,kH,padW,padH,image = helper(inp,kernel)
    output = np.empty(inp.shape,dtype='float32')
    isRGB = len(inp.shape) == 3
    if(isRGB):
        for x in np.arange(0,inW):
            for y in np.arange(0,inH):
                roi = image[y:y+2*padH+1,x:x+2*padW+1,:] # a 3d array
                bMax = np.median(roi[:,:,0]) # aka *u(vert), *v(hor), *sigma
                gMax = np.median(roi[:,:,1])
                rMax = np.median(roi[:,:,2])
                output[y,x] = np.array([bMax,gMax,rMax]) # 3 elem array?
                #output[y,x] = np.median(roi,axis=2) # 3 elem array?
    else:
        for x in np.arange(0,inW):
            for y in np.arange(0,inH):
                roi = image[y:y+2*padH+1,x:x+2*padW+1] # a 3d array
                output[y,x] = np.median(roi)
    return output

def erosion(inp,kernSize,it=1):
    full = kernSize[0]*kernSize[1]
    kernel = np.ones(kernSize)
    sums = cv2.filter2D(inp,5,kernel) # the mins are 0
    final = np.where(sums==full,sums,0)
    final = np.where(final!=full,final,1)
    return final.astype('uint8') # if not uint8, WHITE will not be filtered out by applymask
    '''
    kernel = np.empty((kernSize))
    inW,inH,kW,kH,padW,padH,image = helper(inp,kernel)
    output = np.empty(inp.shape,dtype='float32')
    for i in range(it):
        for x in np.arange(0,inW):
            for y in np.arange(0,inH):
                roi = image[y:y+2*padH,x+1:x+2*padW] # a 3d array
                output[y,x] = np.amin(roi)
    return output
    '''

# currently, we take in opencv images
#(x0,y0) is the top left corner of the mask
# in relation to img
# assume mask is 0
def overlayMask(img,mask,x0,y0,scale=1):
    if(len(img.shape) == 2): # grayscale, then convert to rgb
        img = np.dstack([img,img,img])
    if(x0+mask.shape[1] <= img.shape[1] and y0+mask.shape[0] <= img.shape[0]):
        centering = cvtGray(mask).astype('float32')-230 # converts to gray, removes most
        centering = np.dstack([centering,centering,centering]) # now we only want the negative values to be 1
        binary = ((np.sign(-centering)+1)/2).astype('uint8') # converts to 0 for not show, 1 for show
        upper = binary*mask
        lower = img[y0:y0+mask.shape[0],x0:x0+mask.shape[1]]*(1-binary)
        
        img[y0:y0+mask.shape[0],x0:x0+mask.shape[1]] = upper+lower
    else:
        print(f'failed: {x0+mask.shape[1]}>{img.shape[1]} or {y0+mask.shape[0]} > {img.shape[0]}')
    return img

