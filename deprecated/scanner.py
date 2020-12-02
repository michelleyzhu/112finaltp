import cv2 as cv2
import numpy as np

def empty():
    pass

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
    bgr = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    return bgr

def createTrackbars():
    cv2.namedWindow("trackbars")
    cv2.createTrackbar("hmin","trackbars",0,179,empty)
    cv2.createTrackbar("hmax","trackbars",179,179,empty)
    cv2.createTrackbar("smin","trackbars",0,255,empty)
    cv2.createTrackbar("smax","trackbars",255,255,empty)
    cv2.createTrackbar("vmin","trackbars",0,255,empty)
    cv2.createTrackbar("vmax","trackbars",255,255,empty)

def getContourBox(mask,img):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    largestCnt, largestArea, largestApprox = None, 0, None
    found = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area > 5000 and area > largestArea):
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,peri*0.02,True)
            if(len(approx) == 4):
                found = True
                largestCnt, largestArea, largestApprox = cnt, area, approx
    cv2.putText(mask, f"{largestArea}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
    cv2.putText(img, f"{largestArea}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
    if(found):
        cv2.drawContours(mask, largestCnt, -1, (200,200,200), 4)
        cv2.drawContours(img, largestCnt, -1, (200,200,200), 4)
        rect = cv2.minAreaRect(largestCnt)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(mask, [box], -1, (100,100,100), 4)
        cv2.drawContours(img, [box], -1, (100,100,100), 4)
        return True, largestApprox
    return False, None

widthImg, heightImg = 640,350
blurVal = 3
def preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),3)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDil = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDil,kernel,iterations=1)

    return imgThres
    
    

def reorder(box):
    myPoints = box.reshape((4,2))
    newPoints = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)

    newPoints[0] = myPoints[np.argmin(add)]
    newPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    newPoints[1] = myPoints[np.argmin(diff)]
    newPoints[2] = myPoints[np.argmax(diff)]
    return newPoints

def scan():
    vid = cv2.VideoCapture(0)
    #blurVal = 3
    createTrackbars()

    while(True):
        ret, frame = vid.read()
        img = cv2.resize(frame, (widthImg, heightImg))
        imgCopy = img.copy()
        imgThres = preprocessing(img)
        success, box = getContourBox(imgThres,img)
        rbgimgThres = cv2.cvtColor(imgThres,cv2.COLOR_GRAY2RGB)
        if(success):
            box = reorder(box)
            orig = np.float32([[0,0],[0,widthImg],[0,heightImg],[widthImg,heightImg]])
            matrix = cv2.getPerspectiveTransform(np.float32(box),orig)
            rbgimgThres = cv2.warpPerspective(imgCopy,matrix,(widthImg,heightImg))
        #if(cv2.waitKey(1) == ord('u')):
        #    blurVal += 1
        
        #if(cv2.waitKey(1) == ord('d')):
        #    blurVal -= 1

        #cv2.putText(img,f"{blurVal}",(widthImg//2,heightImg//2),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
        stacked = np.hstack([img, rbgimgThres])
        cv2.imshow("scanner",stacked)
        
        if(cv2.waitKey(1) and 0xFF == ord('q')):
            break


scan()
