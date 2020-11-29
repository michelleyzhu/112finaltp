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
'''
allGraphics = cv2.imread("graphics.jpg")
bombs = cv2.imread("explosions.png")
bigW,bigH = allGraphics.shape[1],allGraphics.shape[0]
lilW,lilH = bigW//10,bigH//10

for row in range(10):
    for col in range(10):
        img = allGraphics[row*lilH:(row+1)*lilH, col*lilW:(col+1)*lilW]
        cv2.imwrite(f'graphics/bw{row}.{col}.jpg', img)

bigW,bigH = bombs.shape[1],bombs.shape[0]
lilW,lilH = bigW//5,bigH//5
for row in range(5):
    for col in range(5):
        img = bombs[row*lilH:(row+1)*lilH, col*lilW:(col+1)*lilW]
        cv2.imwrite(f'graphics/ex{row}.{col}.png', img)
'''
for f in os.listdir('graphics'):
    if(f[0:2] == 'bu'):
        img = cv2.imread(f"graphics/{f}")
        img = cv2.pyrDown(img)
        cv2.imwrite(f'graphics/sp{f}', img)