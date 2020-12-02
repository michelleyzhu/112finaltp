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
    img = cartoonify(img)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = hsv[:,:,1] * 0.5
    hsv = hsv.astype(np.uint8)
    #hsv[:,:,2] = hsv[:,:,2] / ((hsv[:,:,2])/256)**2
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
        centering = cvtGray(mask).astype('float32')-230 # converts to gray, removes most
        centering = np.dstack([centering,centering,centering]) # now we only want the negative values to be 1
        binary = ((np.sign(-centering)+1)/2).astype('uint8') # converts to 0 for not show, 1 for show
        upper = binary*mask
        lower = img[y0:y0+mask.shape[0],x0:x0+mask.shape[1]]*(1-binary)
        
        img[y0:y0+mask.shape[0],x0:x0+mask.shape[1]] = upper+lower
    else:
        print(f'failed: {x0+mask.shape[1]}>{img.shape[1]} or {y0+mask.shape[0]} > {img.shape[0]}')
    return img

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
    else:
        for x in np.arange(0,inW):
            for y in np.arange(0,inH):
                roi = image[y:y+2*padH+1,x:x+2*padW+1] # a 3d array
                output[y,x] = np.median(roi)
    return output

def erosion(inp,kernSize,it=1):
    kernel = np.empty((kernSize))
    inW,inH,kW,kH,padW,padH,image = helper(inp,kernel)
    output = np.empty(inp.shape,dtype='float32')
    for i in range(it):
        for x in np.arange(0,inW):
            for y in np.arange(0,inH):
                roi = image[y:y+2*padH,x+1:x+2*padW] # a 3d array
                output[y,x] = np.amin(roi)
    return output

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

# loosely based off pseudocode of https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
# I have avoided the cv2 calls made in this tutorial and adapted it for rgb channels
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
    blur = genConvolve(img,g5)
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
    bigger = genConvolve(bigger, 4*gup)
    return bigger

def cvtGray(img): # converts bgr to gray
    img = 0.114*img[:,:,0] + 0.587*img[:,:,1] + 0.299*img[:,:,2]
    return img.astype('uint8')

def adapMask(img,kernel,n,C=0):
    if(kernel == 'mean'):
        kernel = np.ones((n,n))/ (n**2)
    elif(kernel == 'gauss'):
        kernel = gaussianGen(n=n)
    thresholds = genConvolve(img,kernel,scale=False)
    diff = img.astype('float32')-thresholds+C
    mask = ((np.sign(diff)+1)/2).astype('uint8')
    above = mask*img
    return mask

def applyMask(img,mask):
    # mask is 1/0 vals
    # img is rgb
    # looking for img on highlighted 1's
    return np.dstack([img[:,:,0]*mask,img[:,:,1]*mask,img[:,:,2]*mask])

def cart(img):
    imgColor = img
    pyrLevels = 3
    biPasses = 7
    print("just copied, now starting:")
    t = time.time()
    for i in range(pyrLevels):
        imgColor = pyrD(imgColor)
    print("ok, pyr'ed down:",time.time()-t)
    t = time.time()
    for i in range(biPasses):
        imgColor = cv2.bilateralFilter(imgColor,d=9,sigmaColor=9,sigmaSpace=7)
    print("bilateraled that:",time.time()-t)
    t = time.time()
    for i in range(pyrLevels):
        imgColor = pyrU(imgColor)
    print("pyred up:",time.time()-t)
    t = time.time()
    
    gray = cvtGray(img)
    blur = medBlur(gray,7)
    print("median blur takes a while",time.time()-t)
    t = time.time()
    myThres = adapMask(blur,'mean',n=9,C=2.5) # adap takes in rgb image, returns 0/1 mask vals
    print("done w/adaptive thresholding",time.time()-t)
    t = time.time()
    #myErode = erosion(myThres,(5,5),it=1) # erosion returns 0/1 mask vals
    print("aaaaaand eroded it",time.time()-t)
    combined = applyMask(img,myThres)
    return combined

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

def convSobel(n):
    smooth = np.array(
        [[ 1, 2, 1 ]], dtype='int'
    )
    smootht = np.array([[1],[2],[1]],dtype='int')
    kernel = smootht*smooth*1/8

    u, s, vh = np.linalg.svd(kernel)
    u1 = u[:,0]
    v1 = vh[0,:]
    sig = s[0]
    kW,kH = kernel.shape[1],kernel.shape[0]
        
    sob3 = smootht*np.array([1,0,-1],dtype='int')

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
    
    return cart(img)

#### TESTING PURPOSES#####
def dumb():
    vid = cv2.VideoCapture(0)
    cv2.namedWindow('recording')
    ret, frame = vid.read()
    windowFrac = 1/2
    minW, minH = int(frame.shape[1]*windowFrac), int(frame.shape[0]*windowFrac)
    w, h = frame.shape[1], frame.shape[0]
    while True:
        ret, frame = vid.read()
        #img = cart(frame)
        img = testConvolve(frame)
        resized = cv2.resize(img,(minW,minH))
        cv2.imshow("recording",resized)
        
        key = cv2.waitKey(1)
        if(key == ord('q')):
            vid.release()
            cv2.destroyAllWindows()
            return

def justIm():
    img = cv2.imread('surprise.jpg')
    #img = cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))
    w,h = img.shape[1],img.shape[0]
    img = img[h//3:2*h//3, w//3:2*w//3]

    cv2.imwrite("orig.jpg",cartoonify(img))
    img = testConvolve(img)
    cv2.imwrite("result.jpg",img)
    print("completed")

def ff(inp):
    # calculate the FFT of the kernel
    # for small array in big array
    # calculate the FFT of the small array
    # element multiply the 2 FFTS
    kernel = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
    bgr = inp[:,:,0] #, img[:,:,1], img[:,:,2]
    
    inW,inH = inp.shape[1],inp.shape[0]
    padW,padH = 3,3
    image = np.ones((inH+2*padH,inW+2*padW),dtype='float32')
    image[padH:inH+padH,padW:inW+padW] = bgr
    
    w,h = image.shape
    kernelimage = np.zeros((w,h))
    kernelimage[:7, :7] = kernel

    fftimage = np.fft.fft(image.astype('float32'))
    fftkernel = np.fft.fft(kernelimage)

    fftblurimage = fftimage*fftkernel

    blurimage = np.fft.ifft(fftblurimage)

    arrMax = np.amax(blurimage)
    arrMin = np.amin(blurimage)
    factor = (arrMax-arrMin+1)/256
    blurimage[:,:] = (blurimage[:,:]-arrMin)/factor
    blurimage = blurimage.astype('uint8')
    return blurimage
    '''
    imgs = []
    for frame in bgr:
        w,h = frame.shape
        kernelimage = np.zeros((w,h))
        kernelimage[:7, :7] = kernel

        fftimage = np.fft.fft(frame.astype('float32'))
        fftkernel = np.fft.fft(kernelimage)

        fftblurimage = fftimage*fftkernel

        blurimage = np.fft.ifft(fftblurimage)
        imgs.append(blurimage)
    
    #cv2.imshow('red',r)
    #cv2.imshow('green',g)
    #cv2.imshow('blue',b)
    #return np.dstack((b,g,r))
    #stacked = np.dstack(tuple(imgs))
    stacked = imgs[0].astype('uint8')
    stacked[:,:] = stacked[:,:]*360//35
    print(np.min(stacked))
    return stacked
    '''
#justIm()