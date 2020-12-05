import math, copy, random, time, string
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
import tkinter.font as tkFont
from PIL import Image
from myCV import *
from cvInterface import *


bubbleRatio = 3.5
headerMarg = 20
pBarLeft, gBarTop, sBarTop, oMarg = 870, 480, 200, 30
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
        self.xMarg, self.yMarg, self.iMarg = 40,50, 10
        self.currX, self.currY = self.xMarg+self.x0, self.yMarg+self.y0
    
    def relocateAll(self):
        self.currX, self.currY = self.xMarg+self.x0, self.yMarg+self.y0
        for drawable in self.drawables:
            self.locateDrawable(drawable)
            
    def locateDrawable(self,drawable):
        self.scaleDrawable(drawable)
        if(self.currX + drawable.w > self.x1):
            self.currX = self.xMarg+self.x0
            self.currY += self.iMarg + drawable.h
        drawable.x, drawable.y = self.currX, self.currY
        self.currX += self.iMarg + drawable.w

    def scaleDrawable(self,drawable):
        drawable.scaleImg(self.scale)
    
    def shouldAccept(self, drawable):
        if(min(self.x1,drawable.x+drawable.w) - max(self.x0,drawable.x) >= 0.5*drawable.w
        and min(self.y1,drawable.y+drawable.h) - max(self.y0,drawable.y) >= 0.5*drawable.h
        and drawable.typ == 'img'):
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
        self.defaultClip = Clip(default,default,0,0,w,h,name="cabbage")
            
        self.shownDrawables = [True]*len(drawables) + [False]*(size - len(drawables))
        self.finalI = len(drawables)-1 # should initialize to -1, empty drawables
        for i in range(len(self.drawables),self.size):
            self.drawables.append(self.defaultClip.copy())
        self.relocateAll()
    
    def clearAll(self):
        self.drawables = []
        
        self.shownDrawables = [False]*self.size
        self.finalI = self.size-1 # should initialize to -1, empty drawables
        for i in range(self.size):
            self.drawables.append(self.defaultClip.copy())
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
                self.drawables[i].editor.updateGraphics()
                self.drawables[i].package()
                return
        
        # if not, then add to the end
        if(self.finalI < self.size-1):
            self.finalI += 1
            self.shownDrawables[self.finalI] = True
        else:
            savedRegion.addDrawable(self.drawables[self.finalI],savedRegion)
        self.drawables[self.finalI] = drawableCopy
        self.relocateAll()
        self.drawables[self.finalI].editor.updateGraphics()
        self.drawables[self.finalI].package()
    
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
    
    def removeDrawableWithoutHiding(self,drawable):
        i = self.drawables.index(drawable)
        # self.drawables[i] = drawable.copy() Why are we copying again?

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
                self.drawables[i] = drawableCopy
                self.shownDrawables[i] = True
                self.finalI = max(i, self.finalI)
                self.relocateAll()
                self.drawables[i].editor.updateGraphics()
                self.drawables[i].package()
                return
        # or if it can't insert anywhere earlier, just insert at end
        self.addDrawable(drawable,savedRegion) #(calls relocateall)
        
    def canMove(self,drawableI):
        return self.shownDrawables[drawableI]

    def finish(self,title=''):
        w,h = self.drawables[0].editor.finalProduct.w,self.drawables[0].editor.finalProduct.h
        final = Image.new('RGB',(2*w+2*self.xMarg+self.iMarg,2*h+2*self.yMarg+self.iMarg),(255,255,255))
        for i in range(self.size):
            if(self.shownDrawables[i]):
                clip = self.drawables[i].editor.finalProduct
                final.paste(clip.img,(self.xMarg+(i%2)*(clip.w+self.iMarg),self.yMarg+(i//2)*(clip.h+self.iMarg)))
        cv = pilToCV(final)
        cv = insertTitle(cv,title)
        final = cvToPIL(cv)
        final.save('myComicStrip.png','PNG')
        
    def draw(self,canvas,littleText=True):
        for i in range(self.size):
            if(self.shownDrawables[i]):
                self.drawables[i].draw(canvas,littleText)

class SavedRegion(Region):
    def __init__(self,name,x,y,w,h,scale,size):
        super().__init__(name,x,y,w,h,scale,size)
        self.xMarg -= 10
        
    def addDrawable(self,drawable,uselessParam): # parameters need to align for for loop in main lol
        drawableCopy = drawable.copy()
        self.drawables.append(drawableCopy)
        self.relocateAll()
    
    def removeDrawableWithoutHiding(self,drawable):
        pass

    def removeDrawable(self,drawable):
        self.drawables.remove(drawable)
        self.relocateAll() # after removing, all x/ys change
        
    def insertDrawable(self,drawable,uselessParam):
        self.addDrawable(drawable,self)
    
    def draw(self,canvas,littleText=True):
        for drawable in self.drawables:
            drawable.draw(canvas,littleText)
        
class EditorRegion(Region):
    def __init__(self,name,img,x,y,w,h,scale=0.55,size=1):
        super().__init__(name,x,y,w,h,scale,size,drawables=[img])
        self.oMarg, self.headerMarg = 50,20
        self.finalProduct = img
        self.filter = 'default'
        self.prevNumGraphics = 1
        self.textUpdated = None
        self.finishedBubble = False
        self.small = tkFont.Font(family='Avenir',size=13)  
        self.textSize = 1.75
        self.bubbleTexts = []
        self.relocateAll()
        self.active = False

    def resizeAll(self):
        for drawable in self.drawables:
            self.scaleDrawable(drawable)
            #drawable.x -= self.x0
            #drawable.y -= self.y0
            # combine images, insert into finalProduct

    def addDrawable(self,drawable,uselessParam): # parameters need to align for for loop in main lol
        if(drawable.origImg != self.drawables[0].origImg):
            drawableCopy = drawable.copy()
            self.drawables.append(drawableCopy)
            self.resizeAll()
        
    def removeDrawable(self,drawable):
        self.drawables.remove(drawable)
        
    def insertDrawable(self,drawable,uselessParam):
        self.addDrawable(drawable,self)
    
    # RETURNS NONE or the center of the bubble(hopefully)
    def bubbleClicked(self,x,y):
        for graphic in self.drawables[1:][::-1]:
            if(graphic.typ != 'bubble'):
                continue
            w,h = int(graphic.img.size[0]*bubbleRatio),int(graphic.img.size[1]*bubbleRatio)
            x0,y0 = graphic.x,graphic.y
            if(x0 < x < x0+w and y0 < y < y0+h):
                return graphic.x + graphic.w*bubbleRatio*self.scale//2,graphic.y + graphic.h*bubbleRatio*self.scale//2,graphic, 1.75
        return None
    
    def insertBubbleText(self,text,x,y,graphic,size):
        for i in range(len(self.bubbleTexts)):
            if(self.bubbleTexts[i][3] == graphic):
                self.bubbleTexts[i] = (text,x,y,graphic,size)
                self.textUpdated = (text,x,y,graphic,size)
                return
        self.bubbleTexts.append((text,x,y,graphic,size))
        self.textUpdated = (text,x,y,graphic,size)
        

    def updateGraphics(self,filterChanged=False):
        # disaster? copy
        tempClip = self.drawables[0]
        # applying filter
        tempImg = tempClip.origImg#.astype('uint8')
        if(self.filter == 'dot'):
            tempImg = halfDotFilter(tempImg,cart(tempImg))
        elif(self.filter == 'pastel'):
            tempImg = pastelFilter(cart(tempImg))
        elif(self.filter == 'default'):
            tempImg = cart(tempImg)
        elif(self.filter == 'sketch'):
            tempImg = cannyFilter(tempImg)
        elif(self.filter == 'vignette'):
            tempImg = vignette(tempImg,cart(tempImg))
        elif(self.filter == 'benday'):
            tempImg = halftone(tempImg)
        success = True
        for graphic in self.drawables[1:]:
            ratio = 1
            if(graphic.typ == 'bubble'):
                ratio = bubbleRatio
            elif(graphic.typ == 'graphic'):
                ratio = 1.5
            scaledGraphic = graphic.img.resize((int(graphic.img.size[0]*ratio),int(graphic.img.size[1]*ratio)))
            mask = pilToCV(scaledGraphic)
            #print(f'graphic,{graphic.x,graphic.y}')
            #print(int((graphic.x-self.x0)/self.scale*1 - self.oMarg),mask.shape,int((graphic.y-self.y0)/self.scale - self.headerMarg))
            tempImg, success = overlayMask(tempImg,mask,int((graphic.x-self.x0)/self.scale*1 - self.oMarg),int((graphic.y-self.y0)/self.scale - self.headerMarg))
            if(not success):
                self.drawables.remove(graphic)
        for message in self.bubbleTexts:
            mess,cX,cY,graphic,size = message
            messages = mess.split('`')
            y = (cY - 30*size*(len(messages))/2)/self.scale# messages = # of lines
            for m in messages:
                x = (cX - 4*size*len(m)/2-75)/self.scale-self.oMarg
                tempImg = insertText(tempImg,f'{m}',(int(x),int(y)),(255,255,255),size)
                y += 25*size # what will the difference be? Check
        tempImg = cvToPIL(tempImg)

        tempClip.img = tempImg # removed copy, disaster?
        tempClip.scaleImg(self.scale)
        self.finalProduct = tempClip.copy() # removed copy, disaster?
        self.finalProduct.img.save('finalProduct.png','PNG')
        return success

    def updateTextSize(self,size):
        self.textSize = size
    def applyFilter(self,filt):
        self.filter = filt
        self.updateGraphics(True)
    def draw(self,canvas,littleText=False):
        if(self.prevNumGraphics != len(self.drawables) or self.finishedBubble):
            self.updateGraphics(False)
            self.prevNumGraphics = len(self.drawables)
            self.finishedBubble = False
        if(self.textUpdated != None):
            mess,cX,cY,graphic,size = self.textUpdated
            messages = mess.replace("`", "\n")
            x,y = 930,650#pBarLeft + oMarg, gBarTop + oMarg
            canvas.create_text(x,y,fill='gray',text=messages,font=self.small,anchor='nw')

        self.finalProduct.draw(canvas,False) # false b/c not drawing little text
        
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
    def draw(self,canvas,littleText=False):
        for drawable in self.drawables:
            drawable.draw(canvas,littleText)
            
def getRandStr():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(8))
    
class Clip():
    def __init__(self,img,unfilt,x,y,w,h,scale=1,name=getRandStr(),typ='img',editor=None,smallBubbleTexts=[],direct=''):
        self.name = name
        self.origImg = unfilt
        self.img = img
        self.x,self.y,self.origW,self.origH = x,y,w,h
        self.typ = typ
        self.scale = scale
        self.getScaledWs()
        self.getCenter()
        self.editor = editor
        self.direct = direct

        # draw rectangle:
        if(typ=='img' and img != None):
            self.img = cvToPIL(outline(pilToCV(self.img),self.w//90))

        self.smallBubbleTexts = smallBubbleTexts
        if(self.editor == []):
            self.editor = EditorRegion(f"editor{getRandStr()}",self,0,headerMarg,w,h)
        
    def package(self):
        # removed a copy..l disaster?
        self.img = self.editor.finalProduct.img
        self.scaleImg(self.scale)
        for message in self.editor.bubbleTexts:
            graphic = message[3]
            x,y = (self.scale*message[1])//0.6 + self.x, (self.scale*(message[2]-100))//0.6 + self.y
            self.smallBubbleTexts.append((message[0],x,y,message[3]))
        
    def __repr__(self): return self.name
    def __eq__(self,clip):
        if(isinstance(clip,Clip)):
            return self.name == clip.name
        return False
    def copy(self):
        return Clip(self.img,self.origImg,self.x,self.y,self.origW,self.origH,self.scale,getRandStr(),typ=self.typ,editor=self.editor,smallBubbleTexts=self.smallBubbleTexts) 
    def getCenter(self): self.cX,self.cY = self.x+self.w//2, self.y+self.h//2
    def scaleImg(self,newScale):
        self.scale = newScale
        self.getScaledWs()
        self.getCenter()
    def getScaledWs(self):
        self.w,self.h = int(self.scale*self.origW),int(self.scale*self.origH)
        self.img = self.img.resize((self.w,self.h))
    
    def draw(self,canvas,littleText=True):
        #if(self.typ == 'img'):
        #    self.img = self.editor.finalProduct.img
        iMarg = 10
        canvas.create_image(self.w//2+self.x, self.h//2+self.y,image=ImageTk.PhotoImage(self.img))
        
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

class ImageButton(Button):
    def __init__(self,label,cX,cY,image,hoverImage=None,swapped=False):
        self.bw,self.bh = image.size
        self.img = image
        self.label = label
        self.hoverImg = hoverImage
        self.bx,self.by = cX - self.bw//2, cY - self.bh//2
        self.hover = False
        self.swapped = swapped
        if(self.swapped):
            self.swapHover()

    def swapHover(self):
        self.hoverImg, self.img = self.img, self.hoverImg
        self.swapped = not self.swapped

    def checkHover(self,x,y):
        clicked = self.isClicked(x,y)
        if(clicked != self.hover): # if changed
            self.hover = clicked
            return True
        return False

    def draw(self,canvas):
        if(not self.hover):
            canvas.create_image(self.bw//2+self.bx, self.bh//2+self.by,image=ImageTk.PhotoImage(self.img))
        else:
            canvas.create_image(self.bw//2+self.bx, self.bh//2+self.by,image=ImageTk.PhotoImage(self.hoverImg))
        
class cvButton(Button):
    def __init__(self,label,x,y,w,h,color='#fe4a49'):
        super().__init__(label,x,y,w,h)
        self.rgb = tuple(int(self.color.strip("#")[i:i+2], 16) for i in (0, 2, 4))
    
    def draw(self,frame):
        cv2.rectangle(frame,(self.bx,self.by),(self.bx+self.bw,self.by+self.bh),self.rgb,-1)
        cv2.putText(frame,f"{self.label}",(self.cX,self.cY+self.bh),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0))
    
    def click(self,frame):
        cv2.rectangle(frame,(300,300),(400,400),(0,255,0),-1)
        