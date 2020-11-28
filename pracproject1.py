import math, copy, random, string, time
import cv2 as cv2
import numpy as np

def empty():
    pass

def captureVid():
    video = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while(True):
        ret, frame = video.read()
        '''
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 200, 500)
        cv2.createTrackbar("hmin", "Trackbars",0,179,empty)
        cv2.createTrackbar("hmax", "Trackbars",179,179,empty)
        cv2.createTrackbar("vmin", "Trackbars",0,255,empty)
        cv2.createTrackbar("vmax", "Trackbars",255,255,empty)
        cv2.createTrackbar("smin", "Trackbars",0,255,empty)
        cv2.createTrackbar("smax", "Trackbars",255,255,empty)

        lower = np.array([cv2.getTrackbarPos("hmin", "Trackbars"),
                        cv2.getTrackbarPos("vmin", "Trackbars"),
                        cv2.getTrackbarPos("smin", "Trackbars")])
        higher = np.array([cv2.getTrackbarPos("hmax", "Trackbars"),
                        cv2.getTrackbarPos("vmax", "Trackbars"),
                        cv2.getTrackbarPos("smax", "Trackbars")])
        '''
        lower = np.array([52,0,0])
        higher = np.array([143,122,255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,lower, higher)
        hsv = cv2.bitwise_and(hsv,hsv, mask=mask)
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        faces = faceCascade.detectMultiScale(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y,w,h),(255,5,5),2)
            gray[y:y+h, x:x+w] = cv2.bitwise_and(hsv, hsv, mask=mask)[y:y+h, x:x+w]
        '''
        
        
        width, height = int(hsv.shape[1]*.4), int(hsv.shape[0]*.4)
        #gray = cv2.resize(gray, (width,height))
        hsv = cv2.resize(hsv, (width,height))
        frame = cv2.resize(frame, (width,height))
        
        hor = np.hstack((hsv, frame))
        cv2.imshow("main", hor)
        if(cv2.waitKey(1) and 0xFF == ord('q')):
            break

    video.release()
    cv.destroyAllWindows()

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def captureImage():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

def getContours(img,frame):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    largest = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(largest == None or largest[1] < area):
            largest = (cnt, area, cv2.arcLength(cnt,True))
        if(area > 500):
            cv2.drawContours(frame, cnt, -1, (255,255,0),3)
    if(largest != None):
        cv2.putText(frame, f"largest: {largest[1], largest[2]}",(50,50), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0))


def colors():
    vid = cv2.VideoCapture(0)

    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

    cv2.namedWindow("trackbars")
    cv2.resizeWindow("trackbars",200,500)
    cv2.createTrackbar("cannyMin","trackbars",0,255, empty)
    cv2.createTrackbar("cannyMax","trackbars",255,255,empty)

    while(True):

        ret, frame = vid.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color = np.zeros((hsv.shape[0],hsv.shape[1],3),np.uint8)
        color[:] = (0,0,0)
        hS, hE, wS, wE = int(hsv.shape[0]/5), int(hsv.shape[0]*2/5), int(hsv.shape[1]/5), int(hsv.shape[1]*2/5)
        color[hS:hE, wS:wE] = (255,0,0)
        hsv = cv2.add(hsv, color)
        
        largestFace = None
        faces = faceCascade.detectMultiScale(hsv)
        for (x,y,w,h) in faces:
            cv2.rectangle(hsv, (x,y),(x+w,y+h),(255,0,255))
            if(largestFace == None or w*h > largestFace[2]*largestFace[3]):
                largestFace = (x,y,w,h)
        
        if(largestFace != None):
            x,y,w,h = largestFace
            cv2.rectangle(color, (x,y),(x+w,y+h),(0,255,255))
            cannyMin = cv2.getTrackbarPos("cannyMin", "trackbars")
            cannyMax = cv2.getTrackbarPos("cannyMax", "trackbars")

            canny = cv2.GaussianBlur(frame, (7,7),1)
            canny = cv2.Canny(canny, cannyMin, cannyMax)
            getContours(canny,frame)
            canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
            hsv[y:y+h, x:x+w] = canny[y:y+h, x:x+w]
            # used to have weird rectangle touching code didn't work lol

        width, height = (int(frame.shape[1]*.4), int(frame.shape[0]*.4))
        frame = cv2.resize(frame,(width,height))
        hsv = cv2.resize(hsv,(width,height))
        color = cv2.resize(color,(width,height))
        stack = np.hstack([frame,hsv,color])
        cv2.imshow("window", stack)
    
        #img = np.zeroes([512,512,3])
        if cv2.waitKey(1)%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

def smiles():
    vid = cv2.VideoCapture(0)

    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")

    cv2.namedWindow("trackbars")
    cv2.resizeWindow("trackbars",200,500)
    cv2.createTrackbar("cannyMin","trackbars",0,255, empty)
    cv2.createTrackbar("cannyMax","trackbars",255,255,empty)

    while(True):
        ret, frame = vid.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        largestFace = None
        faces = faceCascade.detectMultiScale(hsv,1.1,3)
        for (x,y,w,h) in faces:
            cv2.rectangle(hsv, (x,y),(x+w,y+h),(255,0,255))
            if(largestFace == None or w*h > largestFace[2]*largestFace[3]):
                largestFace = (x,y,w,h)
        
        if(largestFace != None):
            x,y,w,h = largestFace
            cv2.rectangle(hsv, (x,y),(x+w,y+h),(0,255,255))
            
        width, height = (int(frame.shape[1]*.4), int(frame.shape[0]*.4))
        frame = cv2.resize(frame,(width,height))
        hsv = cv2.resize(hsv,(width,height))
        stack = np.hstack([frame,hsv])
        cv2.imshow("window", stack)
    
        if cv2.waitKey(1)%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

#smiles()
colors()
#captureImage()