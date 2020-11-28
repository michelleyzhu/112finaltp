import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
from PIL import Image
import os

'''
allGraphics = cv2.imread("graphics.jpg")
bigW,bigH = allGraphics.shape[1],allGraphics.shape[0]
lilW,lilH = bigW//10,bigH//10

for row in range(10):
    for col in range(10):
        img = allGraphics[row*lilH:(row+1)*lilH, col*lilW:(col+1)*lilW]
        cv2.imwrite(f'graphics/{row}.{col}.jpg', img)
'''

allGraphics = cv2.imread("graphics.jpg")
bigW,bigH = allGraphics.shape[1],allGraphics.shape[0]
lilW,lilH = bigW//10,bigH//10

for row in range(10):
    for col in range(10):
        img = allGraphics[row*lilH:(row+1)*lilH, col*lilW:(col+1)*lilW]
        cv2.imwrite(f'graphics/{row}.{col}.jpg', img)