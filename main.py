import math, copy, random, time, string, os
import cv2 as cv2
import numpy as np
from tkinter import *
from cmu_112_graphics import *
from widgets import *

# #fe4a49 • #2ab7ca • #fed766 • #e6e6ea • #f4f4f8
def make2dList(rows, cols):
    return [ ([0] * cols) for row in range(rows) ]

class Studio(App):
    def appStarted(self):
        self.recording = False
        self.vid = None
        self.frame = None
        self.currIm = None
        self.timerDelay = 20
        self.commandHeld = False

        self.doubleClickStarted,self.timeAt = False,time.time()

        self.maxClips = 4
        self.prevRegion = None

        self.pBarLeft = 800
        self.gBarTop = 450

        # margins
        self.oMarg = 30
        self.iMarg = 10
        self.initializeStudio()
        self.importGraphics()
        self.draggedClip = None
        self.vidOut = None
        self.editing = False

        self.studioRegion = StudioRegion("studio",0,0,self.pBarLeft, self.gBarTop,1/4,self.maxClips)
        self.savedRegion = SavedRegion("saved",self.pBarLeft,100,self.width-self.pBarLeft, self.gBarTop-100,0.1,10)
        self.graphicsRegion = GraphicsRegion('graphics',0,self.gBarTop,self.width,self.height-self.gBarTop,0.3,20,self.graphics)
        self.regions = [self.studioRegion, self.savedRegion,self.graphicsRegion]
        

    def initializeStudio(self):
        #self.cartoonGrid = [[None,None],[None,None]]
        l = self.pBarLeft + self.oMarg
        self.buttons = []
        self.recordButt = Button("click to start recording",l,self.oMarg,200,50)
        self.vidButt = Button("video version",l,self.oMarg + 50+self.iMarg,200,50)
        self.buttons.append(self.recordButt)
        self.buttons.append(self.vidButt)

    def removeTempFiles(self,path, suffix='.DS_Store'):
        if path.endswith(suffix):
            print(f'Removing file: {path}')
            os.remove(path)
        elif os.path.isdir(path):
            for filename in os.listdir(path):
                self.removeTempFiles(path + '/' + filename, suffix)

    def importGraphics(self):
        self.graphics = []
        #self.removeTempFiles('graphics')
        for f in os.listdir('graphics'):
            img = self.loadImage(f"graphics/{f}")
            graphicAsClip = Clip(img,0,0,img.size[0],img.size[1],outline=False)
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
    def mousePressed(self,event):
        if(self.recordButt.isClicked(event.x,event.y)):
            self.toggleRecordButton()
            return
        '''
        if(self.vidButt.isClicked(event.x,event.y)):
            self.toggleVidButton()
            return
        '''
        for region in self.regions:
            if(not region.active): continue # if inactive, don't respond to dragging
            for i in range(len(region.drawables)):
                drawable = region.drawables[i]
                if(region.canMove(i) and drawable.isClicked(event.x,event.y)):
                    self.dragX,self.dragY,self.draggedClip = event.x,event.y,drawable
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
        self.currEditorRegion.active = True
        
    def keyPressed(self,event):
        # end editing mode
        if(self.editing and event.key == "Escape"):
            # Todo: package editorregion into image, insert back in previous location
            self.editing = False
            for region in self.regions:
                region.active = True
            self.regions.remove(self.currEditorRegion)
                
    def mouseReleased(self,event):
        if(self.draggedClip != None):
            clip2Add = self.draggedClip.copy()
            if(self.prevRegion.active):
                self.prevRegion.removeDrawable(self.draggedClip)
            for region in self.regions:
                if(region.active and region.shouldAccept(clip2Add)):
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
        cv2.imwrite("savedIm.jpg",self.imgCartoon)

        self.currIm = self.loadImage("savedIm.jpg")
        w, h = self.currIm.size[0], self.currIm.size[1]
        newClip = Clip(self.currIm,0,0,w,h)
        self.studioRegion.addDrawable(newClip,self.savedRegion)
        self.regions.append(newClip.editor)
        
    def cvMouseActed(self,event,x,y,flag,param):
        if(self.filterB.isClicked(x//self.windowFrac,y//self.windowFrac)):
            self.filterB.click(self.imgCartoon)
            self.show()
    

    # cartoonify() inspired by the tutorial
    # https://www.askaswiss.com/2016/01/how-to-create-cartoon-effect-opencv-python.html
    def cartoonify(self,img):
        imgCartoon = img.copy()
        imgColor = imgCartoon.copy()

        pyrLevels = 3
        for i in range(pyrLevels):
            imgColor = cv2.pyrDown(imgColor)
        imgColor = cv2.bilateralFilter(imgColor,d=13,sigmaColor=15,sigmaSpace=15)
        for i in range(pyrLevels):
            imgColor = cv2.pyrUp(imgColor)
        
        imgEdge = cv2.cvtColor(imgCartoon,cv2.COLOR_BGR2GRAY)
        imgEdge = cv2.medianBlur(imgEdge,7)
        imgEdge = cv2.adaptiveThreshold(imgEdge,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=9,C=2.5)
        imgEdge = cv2.erode(imgEdge,(5,5),iterations=3)
        imgCartoon = cv2.bitwise_and(imgColor,imgColor,mask=imgEdge)
        return imgCartoon

    def show(self):
        resized = cv2.resize(self.imgPanel,(self.minW,self.minH))
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
        self.imgCartoon = self.cartoonify(self.frame)
        if(self.vidOut != None):
            self.writeToVid()
        self.imgPanel = self.imgCartoon.copy()
        self.drawPanel(self.imgPanel)
        self.show()
        
    def timerFired(self):
        if(self.recording):
            self.record()
    
    def drawBoard(self,canvas):
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
    Studio(width = 1200, height = 750)

def main():
    playGame()

if __name__ == '__main__':
    main()
