import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
from PIL import Image


# #fe4a49 • #2ab7ca • #fed766 • #e6e6ea • #f4f4f8
class Region():
    def __init__(self,name,x,y,w,h,scale,size,drawables=None):
        self.name = name
        self.size = size
        self.x0, self.y0,self.w,self.h,self.x1,self.y1 = x,y,w,h, x+w, y+h
        self.scale = scale
        self.active = True
        if(drawables == None):
            self.drawables = []
        else:
            self.drawables = drawables

        self.oMarg, self.iMarg = 30, 10
        self.currX, self.currY = self.oMarg+self.x0, self.oMarg+self.y0
    
    def relocateAll(self):
        self.currX, self.currY = self.oMarg+self.x0, self.oMarg+self.y0
        for drawable in self.drawables:
            self.locateDrawable(drawable)
    
    def locateDrawable(self,drawable):
        self.scaleDrawable(drawable)
        if(self.currX + drawable.w > self.x1):
            self.currX = self.oMarg+self.x0
            self.currY += self.iMarg + drawable.h
        drawable.x, drawable.y = self.currX, self.currY
        self.currX += self.iMarg + drawable.w

    def scaleDrawable(self,drawable):
        drawable.scaleImg(self.scale)
    
    def shouldAccept(self, drawable):
        if(min(self.x1,drawable.x+drawable.w) - max(self.x0,drawable.x) >= 0.5*drawable.w
        and min(self.y1,drawable.y+drawable.h) - max(self.y0,drawable.y) >= 0.5*drawable.h
        and drawable.outline): # if outline, then not graphic
            return True
        return False
    
    def canMove(self,drawableI):
        return True
    
    def __eq__(self,region):
        if(isinstance(region,Region)):
            return self.name == region.name
        return False

class StudioRegion(Region):
    def __init__(self,name,x,y,w,h,scale,size,drawables=[]):
        super().__init__(name,x,y,w,h,scale,size)
        default = Image.open("savedIm.jpg")
        w, h = default.size[0]//4, default.size[1]//4
        default = default.resize((w,h))
        defaultClip = Clip(default,0,0,w,h,name="cabbage")
            
        self.shownDrawables = [True]*len(drawables) + [False]*(size - len(drawables))
        self.finalI = len(drawables)-1 # should initialize to -1, empty drawables
        for i in range(len(self.drawables),self.size):
            self.drawables.append(defaultClip.copy())
        self.relocateAll()
    
    # either add to end, or replace the last one
    def addDrawable(self,drawable,savedRegion):
        drawableCopy = drawable.copy()
        
        # first, try to insert in previous holes
        for i in range(self.finalI):
            if(not self.shownDrawables[i]):
                self.drawables[i] = drawableCopy
                self.shownDrawables[i] = True
                self.relocateAll()
                return
        
        # if not, then add to the end
        if(self.finalI < self.size-1):
            self.finalI += 1
            self.shownDrawables[self.finalI] = True
        else:
            savedRegion.addDrawable(self.drawables[self.finalI],savedRegion)
        self.drawables[self.finalI] = drawableCopy
        self.relocateAll()
    
    def removeDrawable(self,drawable):
        i = self.drawables.index(drawable)
        self.shownDrawables[i] = False # do we ever try to remove None?
        self.drawables[i] = drawable.copy()

        # if we're removing the last shown element, we need to update its index
        if(i == self.finalI):
            for shownI in range(self.finalI-1,-1,-1):
                if(self.shownDrawables[shownI] == True):
                    self.finalI = shownI
                    return
            self.finalI = -1
        self.relocateAll()
        
    # insert is checking if it can be added before
    # drawable param has weird x,y that might align with a curr image
    def insertDrawable(self,drawable,savedRegion):
        drawableCopy = drawable.copy()
        for i in range(self.size):
            if(not self.shownDrawables[i] and self.drawables[i].shouldAccept(drawableCopy)):
                # print("trying to insert at ",i)
                self.drawables[i] = drawableCopy
                self.shownDrawables[i] = True
                self.finalI = max(i, self.finalI)
                self.relocateAll()
                return
        # or if it can't insert anywhere earlier, just insert at end
        self.addDrawable(drawable,savedRegion) #(calls relocateall)
        
    def canMove(self,drawableI):
        return self.shownDrawables[drawableI]

    def draw(self,canvas):
        for i in range(self.size):
            if(self.shownDrawables[i]):
                self.drawables[i].draw(canvas)

class SavedRegion(Region):
    def __init__(self,name,x,y,w,h,scale,size):
        super().__init__(name,x,y,w,h,scale,size)
        
    def addDrawable(self,drawable,uselessParam): # parameters need to align for for loop in main lol
        drawableCopy = drawable.copy()
        self.drawables.append(drawableCopy)
        self.relocateAll()
    
    def removeDrawable(self,drawable):
        self.drawables.remove(drawable)
        self.relocateAll() # after removing, all x/ys change
        
    def insertDrawable(self,drawable,uselessParam):
        self.addDrawable(drawable,self)
    
    def draw(self,canvas):
        for drawable in self.drawables:
            drawable.draw(canvas)

class EditorRegion(Region):
    def __init__(self,name,img,x,y,w,h,scale=0.6,size=1):
        super().__init__(name,x,y,w,h,scale,size,drawables=[img])
        self.finalProduct = img.copy()
        self.relocateAll()
        self.active = False
        
    def resizeAll(self):
        for drawable in self.drawables:
            self.scaleDrawable(drawable)
            # combine images, insert into finalProduct

    def addDrawable(self,drawable,uselessParam): # parameters need to align for for loop in main lol
        drawableCopy = drawable.copy()
        self.drawables.append(drawableCopy)
        self.resizeAll()
        
    def removeDrawable(self,drawable):
        self.drawables.remove(drawable)
        
    def insertDrawable(self,drawable,uselessParam):
        self.addDrawable(drawable,self)
    
    def draw(self,canvas):
        for drawable in self.drawables:
            drawable.draw(canvas)
    def canMove(self,i):
        return False

    def shouldAccept(self, drawable):
        if(min(self.x1,drawable.x+drawable.w) - max(self.x0,drawable.x) >= 0.5*drawable.w
        and min(self.y1,drawable.y+drawable.h) - max(self.y0,drawable.y) >= 0.5*drawable.h): # if outline, then not graphic
            return True
        return False


class GraphicsRegion(Region):
    def __init__(self,name,x,y,w,h,scale,size,graphics):
        super().__init__(name,x,y,w,h,scale,size,graphics)
        self.relocateAll()
        
    def addDrawable(self,drawable,uselessParam): # parameters need to align for for loop in main lol
        self.relocateAll()
        pass
    
    def removeDrawable(self,drawable):
        self.relocateAll()
        pass
        
    def insertDrawable(self,drawable,uselessParam):
        self.relocateAll()
        pass
    
    def draw(self,canvas):
        for drawable in self.drawables:
            drawable.draw(canvas)

def getRandStr():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(8))
    
class Clip():
    def __init__(self,img,x,y,w,h,scale=1,name=getRandStr(),outline=True,editor=None):
        # linking editor to self? an attempt
        self.editor = editor
        if(self.editor == None):
            self.editor = EditorRegion(f"editor{getRandStr()}",self,0,0,w,h))
        # end attempt

        self.name = name
        self.origImg = img
        self.x,self.y,self.origW,self.origH = x,y,w,h
        self.outline = outline
        self.scale = scale
        self.getScaledWs()
        self.getCenter()
        
    def __repr__(self):
        return self.name

    def __eq__(self,clip):
        if(isinstance(clip,Clip)):
            return self.name == clip.name
        return False

    def copy(self):
        return Clip(self.origImg.copy(),self.x,self.y,self.origW,self.origH,self.scale,getRandStr(),outline=self.outline,editor=self.editor)

    def getCenter(self):
        self.cX,self.cY = self.x+self.w//2, self.y+self.h//2
    
    def scaleImg(self,newScale):
        self.scale = newScale
        self.getScaledWs()
        self.getCenter()

    def getScaledWs(self):
        self.w,self.h = int(self.scale*self.origW),int(self.scale*self.origH)
        self.img = self.origImg.resize((self.w,self.h))
    
    def draw(self,canvas):
        self.iMarg = 10
        canvas.create_image(self.w//2+self.x, self.h//2+self.y,image=ImageTk.PhotoImage(self.img))
        if(self.outline):
            canvas.create_rectangle(self.x,self.y,self.x+self.w,self.y+self.h,outline='black',width=self.iMarg//2)
        else:
            canvas.create_rectangle(self.x,self.y,self.x+self.w,self.y+self.h,outline='')

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
    
    def shouldAccept(self, drawable):
        if(min(self.x+self.w,drawable.x+drawable.w) - max(self.x,drawable.x) >= 0.5*drawable.w
        and min(self.y+self.h,drawable.y+drawable.h) - max(self.y,drawable.y) >= 0.5*drawable.h):
            return True
        return False

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

