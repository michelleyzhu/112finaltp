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
    return (1-cannySobel(gray))*255

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
    imgColor = adjustBrightness(imgColor,20)
    
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


def vignette(img,output):
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(img)
    biggestFace,maxArea = None,0
    for (x,y,fW,fH) in faces:
        area = fW*fH
        if(area > maxArea):
            biggestFace = (x,y,fW,fH)
            maxArea = area
    x,y,fW,fH = biggestFace
    cX,cY = x+fW//2,y+fH//2
    w,h = img.shape[1],img.shape[0]
    dots = np.ones(img.shape)
    for x in range(0,w,15):
        for y in range(0,h,15):
            r = min(int(math.sqrt((x-cX)**2+(y-cY)**2)/80),9)
            cv2.circle(dots,(x,y),r,(0,0,0),-1)
    return dots*output

def adjustBrightness(img,c):
    return np.maximum(np.minimum(img.astype('float32')+c,255),0).astype('uint8')

def adjustContrast(img,c):
    factor = 259*(255+c)/(255*(259-c))
    return np.maximum(np.minimum(factor*(img.astype('float32')-128)+128,255),0).astype('uint8')

def halfDotFilter(img,output):
    return halfDot(img)*output

def halfDot(img): # passes in bgr image
    gauss = gaussianGen(n=3)
    gray = cvtGray(img)
    gray = cv2.filter2D(gray,3,gauss)
    gray = adjustContrast(gray,80)
    gray = adjustBrightness(gray,30)
    stackedGray = np.dstack([gray,gray,gray])
    hsv = cvtHSV(stackedGray)
    
    radii = (255-hsv[:,:,2])//30 # 1 for saturation, 2 for value. Which one?
    w,h = img.shape[1],img.shape[0]
    dots = np.ones(img.shape)
    for x in range(0,w,15):
        for y in range(0,h,15):
            #cv2.circle(dots,(x,y),radii[y,x],tuple([int(i) for i in img[y,x]]),-1)
            cv2.circle(dots,(x,y),radii[y,x],(0,0,0),-1)
    dotInv = 255-dots
    return dots

def getMatrix(x,y,img,angle):
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
            #print(x,y,'vs',newCoord)
            if(0<=y<img.shape[1] and 0<=x<img.shape[0]):
                result[x,y] = img[newCoord[0],newCoord[1]]
    return result

def halfDotter(cmyk):
    radii = ((255-np.squeeze(cmyk))//70+1).astype('uint8') # 1 for saturation, 2 for value. Which one?
    w,h = cmyk.shape[1],cmyk.shape[0]
    dots = np.ones(cmyk.shape)
    for x in range(0,w,5):
        for y in range(0,h,5):
            cv2.circle(dots,(x,y),radii[y,x],(0,0,0),-1)
    dotInv = 255-dots
    return dots

def halftone(img):
    b,g,r = np.split(img,3,axis=2)

    #inW,inH,kW,kH,padW,padH, = helper(img,np.array((w//3,h//3)))

    black = np.min([255-r,255-g,255-b])
    #black = np.minimum([255-r],np.minimum([255-g],[255-b]))[0]
    c = ((255-r-black)/(255-black))*255
    m = ((255-g-black)/(255-black))*255
    y = ((255-b-black)/(255-black))*255
    #k = np.dstack([black,black,black])
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(img)
    biggestFace,maxArea = None,0
    for (xP,yP,fW,fH) in faces:
        area = fW*fH
        if(area > maxArea):
            biggestFace = (xP,yP,fW,fH)
            maxArea = area
    w,h = img.shape[1],img.shape[0]
    if(biggestFace != None):
        xPos,yPos,fW,fH = biggestFace
        cX,cY = xPos+fW//2,yPos+fH//2
    else:
        cX,cY = w//2,h//2
    mMat = cv2.getRotationMatrix2D((cX,cY),14,1)
    yMat = cv2.getRotationMatrix2D((cX,cY),31,1)
    kMat = cv2.getRotationMatrix2D((cX,cY),44,1)
    mTilt = cv2.warpAffine(m,mMat,(w,h))
    yTilt = cv2.warpAffine(y,yMat,(w,h))
    #kTilt = cv2.warpAffine(k,kMat,(w,h))

    cDotMask = (1-halfDotter(c))*np.full(img.shape,[255,255,0])
    mDots = 1-halfDotter(mTilt)
    yDots = 1-halfDotter(yTilt)
    #kDots = 1-halfDotter(kTilt)
    
    mMat = cv2.getRotationMatrix2D((cX,cY),-15,1)
    yMat = cv2.getRotationMatrix2D((cX,cY),-30,1)
    kMat = cv2.getRotationMatrix2D((cX,cY),-45,1)
    mTilt = cv2.warpAffine(mDots,mMat,(w,h))
    yTilt = cv2.warpAffine(yDots,yMat,(w,h))
    #kTilt = cv2.warpAffine(kDots,kMat,(w,h))
    
    mDotMask = np.dstack([mTilt,mTilt,mTilt])*np.full(img.shape,[255,0,255])
    yDotMask = np.dstack([yTilt,yTilt,yTilt])*np.full(img.shape,[0,255,255])
    #kDotMask = np.dstack[[kTilt,kTilt,kTilt]]*np.full(img.shape,[255,255,255])

    bMask = cDotMask*mDotMask
    gMask = cDotMask*yDotMask
    rMask = mDotMask*yDotMask # either 0,0,0 or 0,0,255

    bPos = np.squeeze(bMask[:,:,0])
    notB = np.where(bPos == 0,bPos,-1) + 1
    notB = np.dstack([notB,notB,notB])
    
    gPos = np.squeeze(gMask[:,:,1])
    notG = np.where(gPos == 0,gPos,-1) + 1
    notG = np.dstack([notG,notG,notG])
    
    rPos = np.squeeze(rMask[:,:,2])
    notR = np.where(rPos == 0,rPos,-1) +1
    notR = np.dstack([notR,notR,notR])
    
    justC = cDotMask*notB*notG
    justY = yDotMask*notG*notR
    justM = mDotMask*notR*notB

    blackPos = np.prod(mDotMask+cDotMask+yDotMask+bMask+rMask+gMask,axis=2) # 0 at non-black, big nums at black
    notAll = np.where(blackPos == 0,blackPos,-1)+1
    notAll = np.dstack([notAll,notAll,notAll])
    #print(np.amax(bMask+gMask+rMask))
    
    justB = np.dstack([bPos,bPos,bPos])*np.full(img.shape,[255,0,0])
    justG = np.dstack([gPos,gPos,gPos])*np.full(img.shape,[0,255,0])
    justR = np.dstack([rPos,rPos,rPos])*np.full(img.shape,[0,0,255])
    justBlack = np.dstack([blackPos,blackPos,blackPos])*np.full(img.shape,[0,0,0])
    
    total = justC+justY+justM+justB+justG+justR # 255*2 = 510 (255,0,255)+(255,255,0) --> (255)
    existPos = np.sum(total,axis=2) # 0 at not-exist, big nums at black
    notExist = np.where(existPos == 0,existPos,-1)+1
    notExist = np.dstack([notExist,notExist,notExist])
    final = np.full(img.shape,[255,255,255])*notExist + total
    
    return total
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
        img = cart(frame)
        resized = cv2.resize(img,(minW,minH))
        cv2.imshow("recording",resized)
        
        key = cv2.waitKey(1)
        if(key == ord('q')):
            vid.release()
            cv2.destroyAllWindows()
            return

def justIm():
    img = cv2.imread('surprise.jpg')
    w,h = img.shape[1],img.shape[0]
    #img = img[h//3:2*h//3, w//3:2*w//3]
    #mat = getMatrix(h//2,w//2,img,90)
    img = cart(img)
    print(img[500:510,500:510])
    #print(img.shape)
    img = cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))
    #cv2.imwrite('orig.jpg',img)
    cv2.imwrite("result.jpg",img)
    print("completed")

#justIm()
#dumb()