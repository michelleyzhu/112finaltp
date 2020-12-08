import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
from PIL import Image
from fakecv import *

def pilToCV(img):
    rgb = img.convert('RGB')
    cvIm = np.array(img)
    bgr = cvIm[:,:,::-1]
    return bgr.astype('uint8')

def cvToPIL(img):
    img = cv2.cvtColor(img.astype('uint8'),cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img
def mirrorImage(img):
    return img[:,::-1,:]

def insertText(img,text,pos,color,size): # size is like 1.75, etc. 
    blackText = np.ones(img.shape)
    cv2.putText(blackText, f"{text}",pos,cv2.FONT_HERSHEY_DUPLEX,size,(0,0,0),thickness=4) # draw thicker black text
    blackText = img*blackText
    whiteText = np.zeros(img.shape)
    cv2.putText(whiteText, f"{text}",pos,cv2.FONT_HERSHEY_DUPLEX,size,color,thickness=2) # draw thinner white text
    return blackText + whiteText
    

# currently, we take in opencv images
#(x0,y0) is the top left corner of the mask
# in relation to img
# assume mask is 0
def overlayMask(img,mask,x0,y0,scale=1):
    if(len(img.shape) == 2): # grayscale, then convert to rgb
        img = np.dstack([img,img,img])
    if(0 <= x0 and x0+mask.shape[1] <= img.shape[1] and 0 <= y0 and y0+mask.shape[0] <= img.shape[0]):
        centering = cvtGray(mask).astype('float32')-230 # converts to gray, removes most
        centering = np.dstack([centering,centering,centering]) # now we only want the negative values to be 1
        binary = ((np.sign(-centering)+1)/2).astype('uint8') # converts to 0 for not show, 1 for show
        upper = binary*mask
        lower = img[y0:y0+mask.shape[0],x0:x0+mask.shape[1]]*(1-binary)
        
        img[y0:y0+mask.shape[0],x0:x0+mask.shape[1]] = upper+lower
    else:
        return img, False
        print(f'failed: {x0+mask.shape[1]}>{img.shape[1]} or {y0+mask.shape[0]} > {img.shape[0]}')
    return img, True

# cartoonify() inspired by the tutorial
# https://www.askaswiss.com/2016/01/how-to-create-cartoon-effect-opencv-python.html
# however, I wrote the opencv methods by hand
def cart(img,C=0,B=20):
    imgColor = img
    pyrLevels = 3
    biPasses = 7
    for i in range(pyrLevels):
        imgColor = pyrD(imgColor)
    for i in range(biPasses):
        imgColor = cv2.bilateralFilter(imgColor,d=9,sigmaColor=9,sigmaSpace=7)
    for i in range(pyrLevels):
        imgColor = pyrU(imgColor)
    imgColor = adjustBrightness(imgColor,B) # should be 20
    imgColor = adjustContrast(imgColor,C)

    gray = cvtGray(img)
    blur = cv2.medianBlur(gray,7)
    #blur = medBlur(gray,7)
    #gauss = gaussianGen(7)
    #blur = cv2.filter2D(gray,3,gauss)
    myThres = adapMask(blur,'mean',n=7,C=2.08) # adap takes in rgb image, returns 0/1 mask vals
    myErode = erosion(myThres,(3,3),it=1) # erosion returns 0/1 mask vals
    combined = applyMask(imgColor,myErode.astype('uint8'))
    return combined

def cannyFilter(img):
    gray = cvtGray(img)
    return (1-cannySobel(gray))*255

def vignette(img,output):
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(img)
    biggestFace,maxArea = None,0
    for (x,y,fW,fH) in faces:
        area = fW*fH
        if(area > maxArea):
            biggestFace = (x,y,fW,fH)
            maxArea = area
    if(biggestFace != None):
        x,y,fW,fH = biggestFace
        cX,cY = x+fW//2,y+fH//2
    else:
        cX,cY = w//2,h//2
    w,h = img.shape[1],img.shape[0]
    dots = np.ones(img.shape)
    for x in range(0,w,15):
        for y in range(0,h,15):
            r = min(int(math.sqrt((x-cX)**2+(y-cY)**2)/80),9)
            cv2.circle(dots,(x,y),r,(0,0,0),-1)
    return dots*output

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
def halfDotter(cmyk):
    radii = ((255-np.squeeze(cmyk))//60+1).astype('uint8') # 1 for saturation, 2 for value. Which one?
    w,h = cmyk.shape[1],cmyk.shape[0]
    dots = np.ones(cmyk.shape)
    for x in range(0,w,6):
        for y in range(0,h,6):
            cv2.circle(dots,(x,y),radii[y,x],(0,0,0),-1)
    dotInv = 255-dots
    return dots

def halftone(origImg):
    inW,inH,kW,kH,padW,padH,img = helper(origImg,np.ones((2*origImg.shape[1]//5,2*origImg.shape[0]//5)))
    img = img.astype('uint8')
    w,h = img.shape[1],img.shape[0]
    b,g,r = np.split(img,3,axis=2)

    black = np.min([255-r,255-g,255-b])
    #black = np.minimum([255-r],np.minimum([255-g],[255-b]))[0]
    c = ((255-r-black)/(255-black))*255
    m = ((255-g-black)/(255-black))*255
    y = ((255-b-black)/(255-black))*255
    #k = np.dstack([black,black,black])
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    cv2.imwrite('result.jpg',img)
    faces = faceCascade.detectMultiScale(img)
    biggestFace,maxArea = None,0
    for (xP,yP,fW,fH) in faces:
        area = fW*fH
        if(area > maxArea):
            biggestFace = (xP,yP,fW,fH)
            maxArea = area
    if(biggestFace != None):
        xPos,yPos,fW,fH = biggestFace
        cX,cY = xPos+fW//2,yPos+fH//2
    else:
        cX,cY = w//2,h//2
    '''mMat = cv2.getRotationMatrix2D((cX,cY),15,1)
    yMat = cv2.getRotationMatrix2D((cX,cY),30,1)
    #kMat = cv2.getRotationMatrix2D((cX,cY),44,1)
    mTilt = cv2.warpAffine(m,mMat,(w,h))
    yTilt = cv2.warpAffine(y,yMat,(w,h))
    #kTilt = cv2.warpAffine(k,kMat,(w,h))'''


    mMat = getMatrix(cX,cY,15)
    yMat = getMatrix(cX,cY,30)
    mTilt = cv2.warpAffine(m,mMat,(w,h))
    yTilt = cv2.warpAffine(y,yMat,(w,h))
    

    mDots = 1-halfDotter(mTilt)
    yDots = 1-halfDotter(yTilt)
    #kDots = 1-halfDotter(kTilt)
    
    #mMat = cv2.getRotationMatrix2D((cX,cY),-15,1)
    #yMat = cv2.getRotationMatrix2D((cX,cY),-30,1)
    mMat = getMatrix(cX,cY,-15)
    yMat = getMatrix(cX,cY,-30)
    #kMat = cv2.getRotationMatrix2D((cX,cY),-45,1)
    mTilt = cv2.warpAffine(mDots,mMat,(w,h))[padH:padH+inH,padW:padW+inW]
    yTilt = cv2.warpAffine(yDots,yMat,(w,h))[padH:padH+inH,padW:padW+inW]
    #kTilt = cv2.warpAffine(kDots,kMat,(w,h))
    
    cDotMask = ((1-halfDotter(c))*np.full(img.shape,[255,255,0]))[padH:padH+inH,padW:padW+inW]
    mDotMask = np.dstack([mTilt,mTilt,mTilt])*np.full(origImg.shape,[255,0,255])
    yDotMask = np.dstack([yTilt,yTilt,yTilt])*np.full(origImg.shape,[0,255,255])
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

    #blackPos = np.prod(mDotMask+cDotMask+yDotMask+bMask+rMask+gMask,axis=2) # 0 at non-black, big nums at black
    #notAll = np.where(blackPos == 0,blackPos,-1)+1
    #notAll = np.dstack([notAll,notAll,notAll])
    
    justB = np.dstack([bPos,bPos,bPos])*np.full(origImg.shape,[255,0,0])
    justG = np.dstack([gPos,gPos,gPos])*np.full(origImg.shape,[0,255,0])
    justR = np.dstack([rPos,rPos,rPos])*np.full(origImg.shape,[0,0,255])
    #justBlack = np.dstack([blackPos,blackPos,blackPos])*np.full(origImg.shape,[0,0,0])
    
    total = justC+justY+justM+justB+justG+justR # 255*2 = 510 (255,0,255)+(255,255,0) --> (255)
    #existPos = np.sum(total,axis=2) # 0 at not-exist, big nums at black
    #notExist = np.where(existPos == 0,existPos,-1)+1
    #notExist = np.dstack([notExist,notExist,notExist])
    #final = np.full(origImg.shape,[255,255,255])*notExist + total
    return total

def outline(img,width):
    bg = np.zeros(img.shape)
    bg[width:img.shape[0]-width,width:img.shape[1]-width] = 1
    return img*bg

def insertTitle(img,title):
    img[40:100,40:500,:] = 200
    img[42:98,42:498,:] = 0
    img = insertText(img,title,(42,95),(255,255,255),1.75)
    return img

#### TESTING PURPOSES#####

def justIm():
    frame = cv2.imread('surprise.jpg')
    #w,h = img.shape[1],img.shape[0]
    #img = img[h//3:2*h//3, w//3:2*w//3]
    #mat = getMatrix(h//2,w//2,img,90)
    #cv2.imwrite('orig.jpg',img)
    #img = insertText(img,'dummy thick',(50,50),(255,255,255))
    #img = outline(img,30)
    #img = np.ones(img.shape)*[238,234,155]
    cv2.imwrite("result.jpg",frame)
    print("completed")

def dumb():
    vid = cv2.VideoCapture(0)
    cv2.namedWindow('recording')
    ret, frame = vid.read()
    windowFrac = 1/2
    minW, minH = int(frame.shape[1]*windowFrac), int(frame.shape[0]*windowFrac)
    w, h = frame.shape[1], frame.shape[0]
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    mouthCascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")
    #mouthCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascades/Mouth.xml") 
    while True:
        ret, frame = vid.read()
        #color = frame
        #frame = cvtGray(frame)
        #frame = cv2.resize(frame,(frame.shape[1]//3,frame.shape[0]//3))
        faces = faceCascade.detectMultiScale(frame)
        largestFace, area = None,0
        for x,y,w,h in faces:
            if(largestFace == None or w*h > area):
                largestFace, area = (x,y,w,h), w*h
        if(largestFace != None and area > 40000):
            x,y,w,h = largestFace
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
            mouths = mouthCascade.detectMultiScale(frame[y+3*h//4:y+h,x:x+w])
            largestMouth, area = None, 0
            for mx,my,mw,mh in mouths:
                if(largestMouth == None or mw*mh > area):
                    largestMouth, area = (mx,my,mw,mh), mw*mh
            if(largestMouth != None):
                mx,my,mw,mh = largestMouth
                cv2.rectangle(frame,(x+mx,y+3*h//4+my),(x+mx+mw,y+3*h//4+my+mh),(0,255,0),5)
        cv2.imshow("recording",frame)

        key = cv2.waitKey(1)
        if(key == ord('q')):
            vid.release()
            cv2.destroyAllWindows()
            return

#justIm()
#dumb()