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
def removeTempFiles(path, suffix='.DS_Store'):
        if path.endswith(suffix):
            print(f'Removing file: {path}')
            os.remove(path)
        elif os.path.isdir(path):
            for filename in os.listdir(path):
                removeTempFiles(path + '/' + filename, suffix)
removeTempFiles('labels')
for f in os.listdir('labels'):
    img = Image.open(f'labels/{f}')
    w,h = img.size
    img = img.resize((w//4,h//4))
    img.save(f'labelsButts/{f}','PNG')
'''
    img = cv2.imread(f"labels/{f}")
    img = cv2.resize(img,(img.shape[0]//10,img.shape[1]//10))
    cv2.imwrite(f'labels/new{f}', img)'''