import math, copy, random, time, string, os
import cv2 as cv2
import numpy as np
#from tkinter import *
from cmu_112_graphics import *
from cvhelpers import *
from fakecv import *
from widgets import *

##### CITATION: see importGraphics() for citation of images #####


# #fe4a49 • #2ab7ca • #fed766 • #e6e6ea • #f4f4f8
class Studio(App):
    def appStarted(self):
        # cv things
        self.recording = False
        self.vid = None
        self.frame = None
        self.vidOut = None
        
        # graphics things
        self.timerDelay = 20
        self.enterText = False
        self.currBubble = ["",0,0,None] # message, x, y
        self.doubleClickStarted, self.timeAt = False,time.time()
        self.draggedClip = None
        self.editing = False
        self.maxClips = 4
        self.prevRegion = None

        # margins
        self.pBarLeft, self.gBarTop = 800, 450
        self.oMarg, self.iMarg = 30, 10
        
        self.importGraphics()
        self.initializeStudio()
        
        
    def initializeStudio(self):
        l = self.pBarLeft + self.oMarg
        self.recordButt = Button("click to start recording",l,  self.oMarg,200,50)
        self.vidButt = Button("video version",l,                self.oMarg + 50,200,50)
        self.default = Button("default cartoon",l,              self.oMarg + 100+self.iMarg,200,50)
        self.dotCartoon = Button("dotted half print filter",l,  self.oMarg + 150+self.iMarg,200,50)
        self.pastel = Button("pastel filter",l,                 self.oMarg + 200+self.iMarg,200,50)
        self.buttons = [self.recordButt, self.vidButt, self.dotCartoon, self.pastel, self.default]
        
        self.studioRegion = StudioRegion("studio",0,0,self.pBarLeft, self.gBarTop,1/4,self.maxClips)
        self.savedRegion = SavedRegion("saved",self.pBarLeft,100,self.width-self.pBarLeft, self.gBarTop-100,0.1,10)
        self.graphicsRegion = GraphicsRegion('graphics',0,self.gBarTop,self.width,self.height-self.gBarTop,0.3,20,self.graphics)
        self.regions = [self.studioRegion, self.savedRegion,self.graphicsRegion]


    # Directly copied from https://www.cs.cmu.edu/~112/notes/notes-recursion-part2.html#removeTempFiles
    def removeTempFiles(self,path, suffix='.DS_Store'):
        if path.endswith(suffix):
            print(f'Removing file: {path}')
            os.remove(path)
        elif os.path.isdir(path):
            for filename in os.listdir(path):
                self.removeTempFiles(path + '/' + filename, suffix)


    # bubbles: 211558518, https://depositphotos.com/211558518/stock-illustration-big-set-empty-speech-bubble.html
    # black/white(graphics/bw): Max Luczynski, https://www.behance.net/gallery/47977047/One-Hundred-hand-drawn-cartoon-and-comic-symbols
    # explosions(graphics/colors): Tartila 20799751, https://www.vectorstock.com/royalty-free-vector/exclamation-texting-comic-signs-on-speech-bubbles-vector-20799751
    def importGraphics(self):
        self.graphics = []
        self.removeTempFiles('graphics')
        for f in os.listdir('graphics/bubbles'):
            img = self.loadImage(f"graphics/bubbles/{f}")
            graphicAsClip = Clip(img,img,0,0,img.size[0],img.size[1],typ='bubble')
            self.graphics.append(graphicAsClip)
        for f in os.listdir('graphics/colors'):
            img = self.loadImage(f"graphics/colors/{f}")
            graphicAsClip = Clip(img,img,0,0,img.size[0],img.size[1],typ='graphic')
            self.graphics.append(graphicAsClip)
        for f in os.listdir('graphics/bw'):
            img = self.loadImage(f"graphics/bw/{f}")
            graphicAsClip = Clip(img,img,0,0,img.size[0],img.size[1],typ='graphic')
            self.graphics.append(graphicAsClip)
        
    def toggleRecordButton(self):
        self.recording = not self.recording
        if(self.recording):
            self.recordButt.label = "click to stop recording"
            self.setupWebcam()
        else:
            self.recordButt.label = "click to start recording"
            self.vid.release()
            cv2.destroyAllWindows()
    '''
    def toggleVidButton(self):
        self.recording = not self.recording
        if(self.recording):
            self.vidButt.label = "click to stop recording"
            self.setupWebcam()
        else:
            self.recordButt.label = "video version"
            self.vid.release()
            cv2.destroyAllWindows()
    '''

    def checkBubbleClicked(self,region,event):
        if(type(region) == EditorRegion):
            coords = region.bubbleClicked(event.x,event.y)
            if(coords != None):
                self.enterText = True
                self.currBubble = ['',coords[0],coords[1],coords[2]]
                return True
        return False

    def mousePressed(self,event):
        if(self.recordButt.isClicked(event.x,event.y)):
            self.toggleRecordButton()
            return
        '''
        if(self.vidButt.isClicked(event.x,event.y)):
            self.toggleVidButton()
            return
        '''
        if(self.editing):
            if(self.dotCartoon.isClicked(event.x,event.y)):
                self.currEditorRegion.applyFilter('dot') # should directly edit the final product
            elif(self.pastel.isClicked(event.x,event.y)):
                self.currEditorRegion.applyFilter('pastel')
            elif(self.default.isClicked(event.x,event.y)):
                self.currEditorRegion.applyFilter('default')
        for region in self.regions:
            if(not region.active): continue # if inactive, don't respond to dragging
            for i in range(len(region.drawables)):
                drawable = region.drawables[i]
                if drawable.typ == 'bubble' and self.checkBubbleClicked(region,event): return # CHECKS BUBBLE CLICKS
                if(region.canMove(i) and drawable.isClicked(event.x,event.y)):
                    self.dragX, self.dragY, self.draggedClip = event.x,event.y,drawable
                    self.prevRegion = region
                    if(region != self.graphicsRegion): # if graphics, double clicking shouldn't do anything
                        if(self.doubleClickStarted):
                            if(time.time() - self.timeAt <= 1):
                                self.openEditor(drawable)
                            self.doubleClickStarted = False
                        else:
                            self.doubleClickStarted = True
                            self.timeAt = time.time()
                    return
                
    def openEditor(self,drawable):
        self.editing = True
        self.currEditorRegion = drawable.editor
        for region in self.regions:
            if(region.name != 'graphics'):
                region.active = False
        drawable.editor.active = True
        
    def keyPressed(self,event):
        # can't escape when entering text
        if(self.editing and not self.enterText and event.key == "Escape"):
            self.editing = False
            for region in self.regions:
                i = 0
                for drawable in region.drawables:
                    if(type(region) != EditorRegion and drawable.editor == self.currEditorRegion and (type(region) == StudioRegion and region.shownDrawables[i])):
                        drawable.package(drawable)
                    i+=1
                if(type(region) != EditorRegion):
                    region.active = True
            self.currEditorRegion.active = False
        if(self.enterText):
            if(event.key == 'Enter'):
                self.enterText = False
                mess, x, y, graphic = tuple(self.currBubble)
                self.currEditorRegion.insertBubbleText(mess,x,y,graphic)
            else:
                newChar = event.key
                if(newChar == 'Space'): newChar = ' '
                mess, x, y,graphic = tuple(self.currBubble)
                if(len(mess)%12 == 0):
                    if(len(mess) != 0 and mess[-1] != 'Space'):
                        mess += '-\n'
                    else:
                        mess += '\n'
                self.currBubble = mess + newChar, x,y,graphic
                
    def mouseReleased(self,event):
        if(self.draggedClip != None): #  and not self.enterText
            clip2Add = self.draggedClip.copy()
            if(self.prevRegion.active):
                self.prevRegion.removeDrawable(self.draggedClip)
            else:
                self.prevRegion.removeDrawableWithoutHiding(self.draggedClip)
            for region in self.regions:
                if(region.active and region.shouldAccept(clip2Add)):
                    if(not (type(region) == EditorRegion and clip2Add.typ == 'img')): # if adding own image to editor, reject!
                        region.insertDrawable(clip2Add,self.savedRegion)
            self.draggedClip = None

    def mouseDragged(self,event):
        clip = self.draggedClip
        if(clip != None):
            clip.move(event.x-self.dragX,event.y-self.dragY)
            self.dragX,self.dragY = event.x,event.y
    
    def setupWebcam(self):
        self.vid = cv2.VideoCapture(0)
        ret, self.frame = self.vid.read(0)
        self.windowFrac = 1/3
        self.minW, self.minH = int(self.frame.shape[1]*self.windowFrac), int(self.frame.shape[0]*self.windowFrac)
        self.w, self.h = self.frame.shape[1], self.frame.shape[0]
        cv2.namedWindow('recording')
        cv2.setMouseCallback('recording',self.cvMouseActed)
        
    def drawPanel(self,img):
        # create panel of buttons, drawn later
        self.filterB = cvButton("pink filter",0,0,self.w//5,self.h//5)
        self.filterB.draw(img)

    def saveSnap(self):
        # the working/showing image is "img" and is a PIL Image object
        # the original stored image will be used to update the image, and is a CV-compatible nparray
        cv2.imwrite("savedIm.jpg",self.imgCartoon)
        currIm = self.loadImage("savedIm.jpg")
        w, h = currIm.size[0], currIm.size[1]
        newClip = Clip(currIm,self.frame,0,0,w,h,editor=[])
        self.studioRegion.addDrawable(newClip,self.savedRegion)
        self.regions.append(newClip.editor)
        
    def cvMouseActed(self,event,x,y,flag,param):
        pass
        #if(self.filterB.isClicked(x//self.windowFrac,y//self.windowFrac)):
        #    self.filterB.click(self.imgCartoon)
        #    self.show()

    def show(self):
        resized = cv2.resize(self.imgCartoon,(self.minW,self.minH))
        cv2.imshow("recording",resized)

    def writeToVid(self):
        self.vidOut.write(self.imgCartoon)

    def processVid(self):
        v = cv2.VideoCapture("vidOut.avi")
        self.faceVid = cv2.VideoWriter("faceVid.avi",cv2.VideoWriter_fourcc("M","J","P","G"),5,(self.w,self.h),True)
        faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
        while(True):
            frameExists, frame = v.read()
            if(not frameExists):
                self.faceVid.release()
                break

            faces = faceCascade.detectMultiScale(frame)
            largestFace, area = None,0
            for x,y,w,h in faces:
                if(largestFace == None or w*h > area):
                    largestFace, area = (x,y,w,h), w*h
            if(largestFace != None and area > 80000):
                x,y,w,h = largestFace
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
                self.faceVid.write(frame)
            '''
            else:
                failure = np.zeros((self.w,self.h),dtype=np.uint8)
                cv2.putText(failure,"failure!",(self.w//2,self.h//2),cv2.FONT_HERSHEY_COMPLEX,-1,(255,0,0))
                cv2.imwrite("recording",failure)
                self.faceVid.write(failure)
            '''

    def record(self):
        key = cv2.waitKey(1)
        if(key == ord('q')):
            self.recording = False
            self.vid.release()
            cv2.destroyAllWindows()
            return
        elif(key == ord('s')):
            self.saveSnap()
            self.recording = False
            self.vid.release()
            cv2.destroyAllWindows()
            return
        elif(key == ord('v')):
            self.vidOut = cv2.VideoWriter('vidOut.avi',cv2.VideoWriter_fourcc('M','J','P','G'),5,(self.w,self.h),True)
        elif(key == ord('d')):
            self.vidOut.release()
            self.vidButt.label = "video captured!"
            self.processVid()
        
        ret, self.frame = self.vid.read()
        self.imgCartoon = cart(self.frame) #AHHHH
        if(self.vidOut != None):
            self.writeToVid()
        #self.imgPanel = self.imgCartoon.copy()
        #self.imgPanel = self.imgCartoon
        #self.drawPanel(self.imgPanel)
        self.show()
        
    def timerFired(self):
        if(self.recording):
            self.record()
    
    def drawBoard(self,canvas):
        if(self.editing):
            self.currEditorRegion.draw(canvas,False)
            self.graphicsRegion.draw(canvas,False)
        else:
            for region in self.regions:
                if(region.active):
                    region.draw(canvas)
            
    def drawControlPanel(self,canvas):
        canvas.create_line(self.pBarLeft,0,self.pBarLeft,self.gBarTop,fill='black',width=10)
        for butt in self.buttons:
            butt.draw(canvas)

    def drawGraphicsScrollbar(self,canvas):
        canvas.create_line(0,self.gBarTop,self.width,self.gBarTop,fill='black',width=10)
        pass

    def redrawAll(self, canvas):
        self.drawControlPanel(canvas)
        self.drawGraphicsScrollbar(canvas)
        self.drawBoard(canvas)
        
        
def playGame():
    Studio(width = 1200, height = 800)

def main():
    playGame()

if __name__ == '__main__':
    main()
