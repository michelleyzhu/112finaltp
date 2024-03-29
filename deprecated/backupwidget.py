import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *

# #fe4a49 • #2ab7ca • #fed766 • #e6e6ea • #f4f4f8

class Region():
    def __init__(self,x,y,w,h,drawables,scale):
        self.x0, self.y0,self.w,self.h,self.x1,self.y1 = x,y,w,h, x+w, y+h
        self.scale = scale
        self.drawables = drawables
        self.oMarg, self.iMarg = 30, 10
        self.scaleDrawables()
    
    def scaleDrawables(self):
        x, y = self.oMarg, self.oMarg
        for drawable in self.drawables:
            newW, newH = self.scale*drawable.w, self.scale*drawable.h
            if(x + newW > self.x1):
                x = self.oMarg
                y += self.iMarg + newH
            drawable.updateXYWH(self.x0+x,self.self.y0+y,newW,newH)
            x += self.iMarg + newW
        
    def draw(self,canvas):
        for drawable in self.drawables:
            drawable.draw(canvas)

class Clip():
    def __init__(self,img,x,y,w,h):
        self.img = img
        self.x, self.y,self.w,self.h = x,y,w,h
        self.getCenter()
    
    def getCenter(self):
        self.cX,self.cY = self.x+self.w//2, self.y+self.h//2
    
    def draw(self,canvas):
        self.iMarg = 10
        canvas.create_image(self.w//2+self.x, self.h//2+self.y,image=ImageTk.PhotoImage(self.img))
        canvas.create_rectangle(self.x,self.y,self.x+self.w,self.y+self.h,outline='black',width=self.iMarg//2)

    def click(self):
        canvas.create_text(self.cX,self.cY+self.bh,text=f"clicked??")

    def isClicked(self,x,y):
        if(self.x < x < self.x+self.w and self.y < y < self.y+self.h):
            return True
        return False
    
    def move(self,dx,dy):
        self.x += dx
        self.y += dy
        self.getCenter()

class Button():
    def __init__(self,label,x,y,w,h,color='#fe4a49'):
        self.bx, self.by = (x,y)
        self.bw,self.bh = (w,h)
        self.label = label
        self.getCenter()
        self.color = color

    def getCenter(self):
        self.cX,self.cY = self.bx+self.bw//2, self.by+self.bh//2
    
    def draw(self,canvas):
        canvas.create_rectangle(self.bx,self.by,self.bx+self.bw,self.by+self.bh,fill=self.color)
        canvas.create_text(self.cX,self.cY,text=f"{self.label}")

    def click(self):
        canvas.create_text(self.cX,self.cY+self.bh,text=f"{self.label}")

    def isClicked(self,x,y):
        if(self.bx < x < self.bx+self.bw and self.by < y < self.by+self.bh):
            return True
        return False

class cvButton(Button):
    def __init__(self,label,x,y,w,h,color='#fe4a49'):
        super().__init__(label,x,y,w,h)
        self.rgb = tuple(int(self.color.strip("#")[i:i+2], 16) for i in (0, 2, 4))
    
    def draw(self,frame):
        cv2.rectangle(frame,(self.bx,self.by),(self.bx+self.bw,self.by+self.bh),self.rgb,-1)
        cv2.putText(frame,f"{self.label}",(self.cX,self.cY+self.bh),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0))
    
    def click(self,frame):
        cv2.rectangle(frame,(300,300),(400,400),(0,255,0),-1)
        #cv2.putText(frame,"clicking",(50,50),cv2.FONT_HERSHEY_COMPLEX,9,(255,0,0))




### Backup main.py
'''
        for row in range(len(self.cartoonGrid)):
            for col in range(len(self.cartoonGrid[row])):
                if(self.cartoonGrid[row][col] == None):
                    self.currIm = self.loadImage("savedIm.jpg")
                    self.currIm = self.scaleImage(self.currIm, 1/4)
                    w, h = self.currIm.size[0], self.currIm.size[1]
                    x, y = self.oMarg + col*(w + self.iMarg), self.oMarg + row*(h + self.iMarg)
                    newClip = Clip(self.currIm,x,y,w,h)
                    #self.cartoonGrid[row][col] = 
                    self.studioRegion.drawables.append(newClip)
                    return
        '''


'''
        for row in range(len(self.cartoonGrid)):
            for col in range(len(self.cartoonGrid[row])):
                img = self.cartoonGrid[row][col]
                if(img != None):
                    img.draw(canvas)
        '''

default = Image.open("savedIm.jpg")
w, h = default.size[0]//4, default.size[1]//4
default = default.resize((w,h))
defaultClip = Clip(default,0,0,w,h,name="cabbage")
            

            '''
        tempClip = self.drawables[0].copy()
        tempImg = pilToCV(tempClip.origImg)
        for graphic in self.drawables[1:]:
            mask = pilToCV(graphic.img)
            tempImg = overlayMask(tempImg,mask,int((graphic.x-self.x0)/(self.scale*1))-self.oMarg,int((graphic.y-self.y0)/(self.scale*1))-self.oMarg)
        tempImg = cvToPIL(tempImg)
        tempClip.origImg = tempImg.copy()
        tempClip.scaleImg(self.scale)
        tempClip.draw(canvas)
        '''
        '''
        tempImg = self.origImg.copy()
        tempImg = pilToCV(tempImg)
        for graphic in self.editor.drawables[1:]:
            mask = pilToCV(graphic.img)
            tempImg = overlayMask(tempImg,mask,int((graphic.x-self.editor.x0)/(self.editor.scale*1))-self.editor.oMarg,int((graphic.y-self.editor.y0)/(self.editor.scale*1))-self.editor.oMarg)
            print("packing drawable at",graphic.name)
        tempImg = cvToPIL(tempImg)
        self.origImg = tempImg
        '''


tempClip = self.drawables[0].copy()
        tempImg = pilToCV(tempClip.origImg)
        for graphic in self.drawables[1:]:
            ratio = 1
            if(graphic.typ == 'bubble'):
                ratio = 5
            elif(graphic.typ == 'graphic'):
                ratio = 1.5
            scaledGraphic = graphic.img.resize((int(graphic.img.size[0]*ratio),int(graphic.img.size[1]*ratio)))
            mask = pilToCV(scaledGraphic)
            tempImg = overlayMask(tempImg,mask,int((graphic.x-self.x0)/(self.scale*1))-self.oMarg,int((graphic.y-self.y0)/(self.scale*1))-self.oMarg)
        tempImg = cvToPIL(tempImg)
        tempClip.origImg = tempImg.copy()
        tempClip.scaleImg(self.scale)
        tempClip.draw(canvas)
        for message in self.bubbleTexts:
            canvas.create_text(message[1],message[2]-30,fill='blue',text=message[0],font='helvetica 20 bold')



'''
    def package(self):
        tempImg = self.origImg.copy()
        tempImg = pilToCV(tempImg)
        for graphic in self.editor.drawables[1:]:
            ratio = 1
            if(graphic.typ == 'bubble'):
                ratio = bubbleRatio
            elif(graphic.typ == 'graphic'):
                ratio = 2
            scaledGraphic = graphic.img.resize((int(graphic.img.size[0]*ratio*self.editor.scale),int(graphic.img.size[1]*ratio*self.editor.scale)))
            mask = pilToCV(scaledGraphic)
            tempImg = overlayMask(tempImg,mask,int((graphic.x-self.editor.x0)/(self.editor.scale*1))-self.editor.oMarg,int((graphic.y-self.editor.y0)/(self.editor.scale*1))-self.editor.oMarg)
        for message in self.editor.bubbleTexts:
            x,y = int(message[1]/self.editor.scale)-200, int((message[2]-80)/self.editor.scale)
            i = 0
            for line in message[0].split('\n'):
                cv2.putText(tempImg, line, (x, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),thickness=3)
                y += 30
                i+=1
        tempImg = cvToPIL(tempImg)
        #self.origImg = tempImg
        self.origImg = tempImg.copy()
        '''