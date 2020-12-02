import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
from PIL import Image
from fakecv import *

def dumb():
    vid = cv2.VideoCapture(0)
    cv2.namedWindow('recording')
    ret, frame = vid.read()
    windowFrac = 1/3
    minW, minH = int(frame.shape[1]*windowFrac), int(frame.shape[0]*windowFrac)
    w, h = frame.shape[1], frame.shape[0]
        
    while True:
        ret, frame = vid.read()
        #img = cart(frame)
        img = pastelFilter(frame)
        resized = cv2.resize(img,(minW,minH))
        cv2.imshow("recording",resized)
        
        key = cv2.waitKey(1)
        if(key == ord('q')):
            vid.release()
            cv2.destroyAllWindows()
            return
dumb()