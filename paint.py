import cv2 as cv2
import numpy as np


def testForColor(img):
    hmin = cv2.getTrackbarPos("hmin","trackbars")
    hmax = cv2.getTrackbarPos("hmax","trackbars")
    smin = cv2.getTrackbarPos("smin","trackbars")
    smax = cv2.getTrackbarPos("smax","trackbars")
    vmin = cv2.getTrackbarPos("vmin","trackbars")
    vmax = cv2.getTrackbarPos("vmax","trackbars")
    
    lower = np.array([hmin,smin,vmin])
    upper = np.array([hmax,smax,vmax])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower,upper)
    return cv2.bitwise_and(hsv,hsv,mask=mask)


def getContours(mask,filt):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    largest = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area > 500 and (largest == None or largest[1] < area)):
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.drawContours(filt,cnt,-1,(255,255,0),10)
            cv2.rectangle(filt,(x,y),(x+w,y+h),(255,100,100),3)
            largest = cnt, area, (x,y,w,h)
    if(largest != None):
        x,y,w,h = largest[2]
        return x+w//2, y
    
rbg = [[155,115,155,179,255,255],
        [102,98,163,154,255,255],
        [50,93,71,92,226,208]]
    
def drawHighlight(img,centerList):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    w,h = int(hsv.shape[1]*.2),int(hsv.shape[0]*.2)
    
    #stackList = []
    for color in rbg:
        mask = cv2.inRange(hsv,np.array(color[:3]),np.array(color[3:]))
        filtered = cv2.bitwise_and(hsv,hsv,mask=mask)
        center = getContours(mask,filtered)
        if(center != None):
            x,y = center
            c = (0,0,0)
            if(color[0] == 155):
                c = (0,0,255)
            elif(color[0] == 102):
                c = (255,0,0)
            elif(color[0] == 50):
                c = (0,255,0)
            centerList.append((x,y,c))
        #filtered = cv2.resize(filtered,(w,h))
        #cv2.rectangle(filtered,(0,0),(filtered.shape[1],filtered.shape[0]),(255,255,0))
        #stackList.append(filtered)

    #return stackList


def empty():
    pass

def createTrackbars():
    cv2.namedWindow("trackbars")
    cv2.createTrackbar("hmin","trackbars",0,179,empty)
    cv2.createTrackbar("hmax","trackbars",179,179,empty)
    cv2.createTrackbar("smin","trackbars",0,255,empty)
    cv2.createTrackbar("smax","trackbars",255,255,empty)
    cv2.createTrackbar("vmin","trackbars",0,255,empty)
    cv2.createTrackbar("vmax","trackbars",255,255,empty)

def runWebcam():
    vid = cv2.VideoCapture(0)
    #createTrackbars()
    centerList = []
    while(True):
        ret, frame = vid.read()
        #filtered = testForColor(frame)
        #stackList = drawHighlight(frame,centerList)
        drawHighlight(frame,centerList)
        #stacked = np.hstack(stackList)
        for cX,cY,c in centerList:
            cv2.circle(frame,(cX,cY),25,c,-1)
        #cv2.imshow("checking for color",stacked)
        frame = cv2.resize(frame,(frame.shape[1]//3,frame.shape[0]//3))
        cv2.imshow("summation",frame)
        if(cv2.waitKey(1) and 0xFF == ord('q')):
            break

runWebcam()