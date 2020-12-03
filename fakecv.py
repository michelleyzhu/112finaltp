import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
from PIL import Image
from cvhelpers import *

#Avoid using loops in Python as much as possible, especially double/triple loops etc. They are inherently slow.
#Vectorize the algorithm/code to the maximum extent possible, because Numpy and OpenCV are optimized for vector operations.
#Exploit the cache coherence.
#Never make copies of an array unless it is necessary. Try to use views instead. Array copying is a costly operation.
def pastelFilter(img):
    '''
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
    '''
    '''
    imgColor = img.copy() # avoiding copying?

    pyrLevels = 3
    biPasses = 7
    for i in range(pyrLevels):
        imgColor = cv2.pyrDown(imgColor)
    for i in range(biPasses):
        imgColor = cv2.bilateralFilter(imgColor,d=9,sigmaColor=9,sigmaSpace=7)
    for i in range(pyrLevels-1):
        imgColor = cv2.pyrUp(imgColor)
    '''

    hsv = cvtHSV(img)
    hsv[:,:,1] = hsv[:,:,1]*.25# s lower
    hsv[:,:,2] = 150 + hsv[:,:,2]/10 # v higher
    
    #hsv[:,:,1] = hsv[:,:,1] * 0.5
    #hsv[:,:,2] = hsv[:,:,2] / ((hsv[:,:,2])/256)**2
    hsv = hsv.astype('uint8')
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return rgb

def dotFilter(img):
    pass

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
    
def cannyFilter(img):
    gray = cvtGray(img)
    return cannySobel(gray)

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

def cart(img):
    imgColor = img
    pyrLevels = 3
    biPasses = 7
    for i in range(pyrLevels):
        imgColor = pyrD(imgColor)
    for i in range(biPasses):
        imgColor = cv2.bilateralFilter(imgColor,d=9,sigmaColor=9,sigmaSpace=7)
    for i in range(pyrLevels):
        imgColor = pyrU(imgColor)
    
    gray = cvtGray(img)
    blur = cv2.medianBlur(gray,7)#medBlur(gray,7)
    #gauss = gaussianGen(7)
    #blur = cv2.filter2D(gray,3,gauss)
    #myThres = cannySobel(gray,maxVal=40)
    myThres = adapMask(blur,'mean',n=7,C=2.08) # adap takes in rgb image, returns 0/1 mask vals
    myErode = erosion(myThres,(3,3),it=1) # erosion returns 0/1 mask vals
    combined = applyMask(imgColor,myErode.astype('uint8'))
    
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

#### TESTING PURPOSES#####
def dumb():
    vid = cv2.VideoCapture(0)
    cv2.namedWindow('recording')
    ret, frame = vid.read()
    windowFrac = 1/2
    minW, minH = int(frame.shape[1]*windowFrac), int(frame.shape[0]*windowFrac)
    w, h = frame.shape[1], frame.shape[0]
    #while True:
    ret, frame = vid.read()
    img = cart(frame)
    resized = cv2.resize(img,(minW,minH))
    cv2.imshow("recording",resized[...,::-1])
    cv2.imwrite('result.jpg',resized)
    
    key = cv2.waitKey(1)
    if(key == ord('q')):
        vid.release()
        cv2.destroyAllWindows()
        return

def justIm():
    img = cv2.imread('surprise.jpg')
    #img = cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))
    #w,h = img.shape[1],img.shape[0]
    #img = img[h//3:2*h//3, w//3:2*w//3]

    cv2.imwrite("result.jpg",cart(img))
    cv2.imwrite('orig.jpg',cartoonify(img))
    print("completed")
    
def fakeMedianBlur(img,n):
    meanKern = np.ones((n,n)) / (n*n)
    sharpKern = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    gaussKern = gaussianGen(n)
    meanBlur = cv2.filter2D(img,3,meanKern)
    sharpen = cv2.filter2D(img,3,sharpKern)
    gauss = cv2.filter2D(cvtGray(img),3,gaussKern)
    
    return gauss

justIm()
#dumb()