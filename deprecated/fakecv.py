import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
from PIL import Image
from cvhelpers import *

def cannySobel(gray,n=5,minVal=40,maxVal=75):
    gauss = gaussianGen(n)
    
    imgBlur = cv2.filter2D(gray,3,gauss)
    #imgBlur = genConvolve(gray,gauss)
    
    sobelX = np.array([[-1, 0, 1,],[-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array([[-1, -2, -1,],[0,0,0], [1,2,1]])
    #gX = genConvolve(imgBlur,sobelX)
    #gY = genConvolve(imgBlur,sobelY)
    gX = cv2.filter2D(imgBlur,3,sobelX)
    gY = cv2.filter2D(imgBlur,3,sobelY)
    
    grad = np.sqrt(gX**2 + gY**2)
    grad = np.nan_to_num(grad)
    theta = np.arctan(gY/gX) # -pi/2, pi/2: 0-pi/4, pi/4-pi/2,pi/2-3pi/2,3pi/2-pi mod pi/4, 
    # (theta + pi/2) // pi/4: 0 = hori(0,1/0,-1), 1 = (1,1/-1,-1), 2 = vert(1,0/-1,0), 3 = (1,-1/-1,1)
    direct = (theta + np.pi)//1
    direct = np.nan_to_num(direct)
    suppressed = grad*nonMaxSuppress(grad,direct)
    
    strong = (np.sign(grad-maxVal)+1)//2 # 1 if in, 0 if out
    invStrong = 1-strong*grad
    weakAndStrong = (np.sign(grad-minVal)+1)//2 # 1 if above min, 0 if out
    middle = (strong+weakAndStrong)%2 #1 if in the middle, 0 if out
    
    andKern = np.ones((3,3))
    connected = cv2.filter2D(strong,5,andKern)
    connected = np.where(connected == 0,connected,1)
    acceptedWeaks = middle*connected
    final = middle+strong
    return 1-final

# grad is 1 channel
def nonMaxSuppress(grad,direct):
    # copy from img to output
    kernel = np.zeros((3,3)) # just for padding sake
    inW,inH,kW,kH,padW,padH,padGrad = helper(grad,kernel)

    nw = padGrad[:inH,:inW]
    n = padGrad[:inH,padW:padW+inW]
    ne = padGrad[:inH,2*padW:]
    sw = padGrad[2*padH:,:inW]
    s = padGrad[2*padH:,padW:padW+inW]
    se = padGrad[2*padH:,2*padW:]
    w = padGrad[padH:padH+inH,:inW]
    e = padGrad[padH:padH+inH,2*padW:]

    # 1 for local maximum than that direction, 0 not
    horiMask = (np.sign(grad-w)+np.sign(grad-e))//2 # both positive --> 2 --> 1, all else --> 0
    vertMask = (np.sign(grad-s)+np.sign(grad-n))//2
    posMask = (np.sign(grad-ne)+np.sign(grad-sw))//2
    negMask = (np.sign(grad-nw)+np.sign(grad-se))//2
    
    dHori = np.where(direct == 1,direct,0) #-->
    dPos = np.where(direct == 2,direct,0) # />
    dVert = np.where(direct == 3,direct,0) # |^
    dNeg = np.where(direct == 4,direct,0) # \^
    
    mask = (dHori*horiMask+dVert*vertMask+dPos*posMask+dNeg*negMask)//4
    return mask

# loosely based off pseudocode of https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
# I have avoided the cv2 calls made in this tutorial, adapted it for rgb channels, 
# developed my own border padding, scaling and optimized the kernel through SVD
def genConvolve(inp,kernel,scale=True):
    u1,v1,sig = svdKern(kernel)

    inW,inH,kW,kH,padW,padH,image = helper(inp,kernel)
    output = np.empty(inp.shape,dtype='float32')
    isRGB = len(inp.shape) == 3
    if(isRGB):
        for x in np.arange(0,inW):
            for y in np.arange(0,inH):
                roi = image[y:y+2*padH+1,x:x+2*padW+1,:] # a 3d array
                roi1 = (roi[:,:,0]*u1*v1*sig).sum() # aka *u(vert), *v(hor), *sigma
                roi2 = (roi[:,:,1]*u1*v1*sig).sum()
                roi3 = (roi[:,:,2]*u1*v1*sig).sum()
                output[y,x] = np.array([roi1,roi2,roi3]) # 3 elem array?
    else:
        for x in np.arange(0,inW):
            for y in np.arange(0,inH):
                roi = image[y:y+2*padH+1,x:x+2*padW+1] # a 3d array
                output[y,x] = (roi*u1*v1*sig).sum()
    if(scale):
        output = scaleRGB(output)
    return output

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

    # deal with 0 division by masking 0-val diffs to arbitrary number(1)
    diffCopy = np.where(diff != 0, diff,1)

    hueB = (60 * ((smallR - smallG) / diffCopy) + 240) % 360
    hueG = (60 * ((smallB - smallR) / diffCopy) + 120) % 360
    hueR = (60 * ((smallG - smallB) / diffCopy) + 360) % 360
    
    hue = (hueB*bMask + hueG*gMask + hueR*rMask)/2
    
    cmaxCopy = np.where(cmax != 0, cmax, 1)
    sat = 255*(diff)/cmaxCopy
    
    return np.dstack([hue,sat,val])
        
def adjustBrightness(img,c):
    return np.maximum(np.minimum(img.astype('float32')+c,255),0).astype('uint8')

def adjustContrast(img,c):
    factor = 259*(255+c)/(255*(259-c))
    return np.maximum(np.minimum(factor*(img.astype('float32')-128)+128,255),0).astype('uint8')

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

def getMatrix(x,y,angle):
    a = math.cos(angle/360*2*math.pi)
    b = math.sin(angle/360*2*math.pi)
    mat = np.array([
        [a,b,(1-a)*x-b*y],
        [-b,a,b*x+(1-a)*y]
    ])
    return mat

# img is a 2d matrix
def transform(img,mat):
    result = np.zeros(img.shape)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            vec = np.array([[x],[y],[1]])
            newCoord = np.matmul(mat,vec).astype('int8')[:,0] #newCOord = [x1,y1]
            if(0<=y<img.shape[1] and 0<=x<img.shape[0]):
                result[x,y] = img[newCoord[0],newCoord[1]]
    return result

