import cv2 as cv2
import numpy as np
from PIL import Image 
def cartoon():
    vid = cv2.VideoCapture(0)
    
    while(True):
        ret, frame = vid.read()
        imgCartoon = frame
        imgColor = imgCartoon.copy()
        
        pyrLevels = 3
        biPasses = 7
        for i in range(pyrLevels):
            imgColor = cv2.pyrDown(imgColor)
        
        imgColor = cv2.bilateralFilter(imgColor,d=13,sigmaColor=15,sigmaSpace=15)
        #imgColor2 = cv2.bilateralFilter(imgColor2,d=3,sigmaColor=150,sigmaSpace=150)
        for i in range(pyrLevels):
            imgColor = cv2.pyrUp(imgColor)
        
        
        imgEdge = cv2.cvtColor(imgCartoon,cv2.COLOR_BGR2GRAY)
        imgEdge = cv2.medianBlur(imgEdge,7)
        imgEdge = cv2.adaptiveThreshold(imgEdge,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=9,C=2.5)
        imgEdge = cv2.erode(imgEdge,(5,5),iterations=3)
        imgCartoon1 = cv2.bitwise_and(imgColor,imgColor,mask=imgEdge)
        
        imgCartoon1 = cv2.resize(imgCartoon1,(frame.shape[1]//2,frame.shape[0]//2))
        
        cv2.imshow("one pass",imgCartoon1)


        if(cv2.waitKey(1) and 0xFF == ord('q')):
            break

cartoon()