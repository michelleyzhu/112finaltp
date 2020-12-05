import math, copy, random, time, string, os
import cv2 as cv2
import numpy as np
#from tkinter import *
from cmu_112_graphics import *
import tkinter.font as tkFont
#from cvhelpers import *
from myCV import *
from cvInterface import *
from widgets import *
from main import *

# Directly copied from https://www.cs.cmu.edu/~112/notes/notes-recursion-part2.html#removeTempFiles
def removeTempFiles(path, suffix='.DS_Store'):
    if path.endswith(suffix):
        print(f'Removing file: {path}')
        os.remove(path)
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            removeTempFiles(path + '/' + filename, suffix)

class SplashScreenMode(Mode):
    def appStarted(mode):
        w,h = mode.width,mode.height
        x,y = w//2,h//2 + 200
        mode.title = mode.loadImage(f"graphics/backgrounds/title.png")
        mode.startButt = ImageButton('start',x,y,Image.open(f'graphics/splashes/start.png'),Image.open(f'graphics/splashes/startHover.png'))
        mode.aboutButt = ImageButton('about',x-200,y,Image.open(f'graphics/splashes/about.png'),Image.open(f'graphics/splashes/aboutHover.png'))
        mode.helpButt = ImageButton('help',x+200,y,Image.open(f'graphics/splashes/tips.png'),Image.open(f'graphics/splashes/tipsHover.png'))
        mode.buttons = [mode.startButt,mode.aboutButt,mode.helpButt]

    def redrawAll(mode, canvas):
        canvas.create_image(mode.width//2, mode.height//2,image=ImageTk.PhotoImage(mode.title))
        for butt in mode.buttons:
            butt.draw(canvas)

    def mouseMoved(mode,event):
        for butt in mode.buttons:
            if(butt.checkHover(event.x,event.y)): return

    def mousePressed(mode,event):
        if(mode.startButt.isClicked(event.x,event.y)):
            mode.app.setActiveMode(mode.app.gameMode)
        elif(mode.aboutButt.isClicked(event.x,event.y)):
            mode.app.setActiveMode(mode.app.aboutMode)
        elif(mode.helpButt.isClicked(event.x,event.y)):
            mode.app.setActiveMode(mode.app.helpMode)
            
class HelpMode(Mode):
    def appStarted(mode):
        w,h = mode.width,mode.height
        x,y = w//2,h//2 - 100
        mode.border = mode.loadImage(f"graphics/backgrounds/border.png")
        mode.text = mode.loadImage(f"graphics/backgrounds/instructions.png")
        mode.backButt = ImageButton('back',x,mode.height-100,Image.open(f'graphics/splashes/back.png'),Image.open(f'graphics/splashes/backHover.png'))
        
    def redrawAll(mode, canvas):
        canvas.create_image(mode.width//2, mode.height//2,image=ImageTk.PhotoImage(mode.border))
        canvas.create_image(mode.width//2, mode.height//2,image=ImageTk.PhotoImage(mode.text))
        mode.backButt.draw(canvas)
    
    def mouseMoved(mode,event):
        mode.backButt.checkHover(event.x,event.y)
    
    def mousePressed(mode,event):
        if(mode.backButt.isClicked(event.x,event.y)):
            mode.app.setActiveMode(mode.app.splashScreenMode)

class AltHelpMode(Mode):
    def appStarted(mode):
        w,h = mode.width,mode.height
        x,y = w//2,h//2 - 100
        mode.border = mode.loadImage(f"graphics/backgrounds/border.png")
        mode.text = mode.loadImage(f"graphics/backgrounds/instructions.png")
        mode.backButt = ImageButton('back',x,mode.height-100,Image.open(f'graphics/splashes/back.png'),Image.open(f'graphics/splashes/backHover.png'))
        
    def redrawAll(mode, canvas):
        canvas.create_image(mode.width//2, mode.height//2,image=ImageTk.PhotoImage(mode.border))
        canvas.create_image(mode.width//2, mode.height//2,image=ImageTk.PhotoImage(mode.text))
        mode.backButt.draw(canvas)
    
    def mouseMoved(mode,event):
        mode.backButt.checkHover(event.x,event.y)
    
    def mousePressed(mode,event):
        if(mode.backButt.isClicked(event.x,event.y)):
            mode.app.setActiveMode(mode.app.gameMode)
        
class AboutMode(Mode):
    def appStarted(mode):
        w,h = mode.width,mode.height
        x,y = w//2,h//2 - 100
        mode.border = mode.loadImage(f"graphics/backgrounds/border.png")
        mode.text = mode.loadImage(f"graphics/backgrounds/aboutApp.png")
        mode.backButt = ImageButton('back',x,mode.height-100,Image.open(f'graphics/splashes/back.png'),Image.open(f'graphics/splashes/backHover.png'))
        
    def redrawAll(mode, canvas):
        canvas.create_image(mode.width//2, mode.height//2,image=ImageTk.PhotoImage(mode.border))
        canvas.create_image(mode.width//2, mode.height//2,image=ImageTk.PhotoImage(mode.text))
        mode.backButt.draw(canvas)
    
    def mouseMoved(mode,event):
        mode.backButt.checkHover(event.x,event.y)
    
    def mousePressed(mode,event):
        if(mode.backButt.isClicked(event.x,event.y)):
            mode.app.setActiveMode(mode.app.splashScreenMode)
