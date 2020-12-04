import math, copy, random, time, string, os
import cv2 as cv2
import numpy as np
#from tkinter import *
from cmu_112_graphics import *
import tkinter.font as tkFont
from cvhelpers import *
from fakecv import *
from widgets import *
# Directly copied from https://www.cs.cmu.edu/~112/notes/notes-recursion-part2.html#removeTempFiles
def removeTempFiles(path, suffix='.DS_Store'):
    if path.endswith(suffix):
        print(f'Removing file: {path}')
        os.remove(path)
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            removeTempFiles(path + '/' + filename, suffix)


##### CITATION: see importGraphics() for citation of images #####
class SplashScreenMode(Mode):
    def appStarted(mode):
        w,h = mode.width,mode.height
        x,y = w//2,h//2 + 200
        removeTempFiles('splashButts')
        removeTempFiles('bgs')
        mode.title = mode.loadImage(f"bgs/title.png")
        mode.startButt = ImageButton('start',x,y,Image.open(f'splashButts/start.png'),Image.open(f'splashButts/startHover.png'))
        mode.aboutButt = ImageButton('about',x-200,y,Image.open(f'splashButts/about.png'),Image.open(f'splashButts/aboutHover.png'))
        mode.helpButt = ImageButton('help',x+200,y,Image.open(f'splashButts/tips.png'),Image.open(f'splashButts/tipsHover.png'))
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
        mode.border = mode.loadImage(f"bgs/border.png")
        mode.backButt = ImageButton('back',x,mode.height-100,Image.open(f'splashButts/back.png'),Image.open(f'splashButts/backHover.png'))
        
    def redrawAll(mode, canvas):
        canvas.create_text(mode.width//2,mode.height//2,text='hello, instructions here',fill='blue')
        canvas.create_image(mode.width//2, mode.height//2,image=ImageTk.PhotoImage(mode.border))
        mode.backButt.draw(canvas)
    
    def mouseMoved(mode,event):
        mode.backButt.checkHover(event.x,event.y)
    
    def mousePressed(mode,event):
        if(mode.backButt.isClicked(event.x,event.y)):
            mode.app.setActiveMode(mode.app.splashScreenMode)
        
class AboutMode(Mode):
    def appStarted(mode):
        w,h = mode.width,mode.height
        x,y = w//2,h//2 - 100
        mode.border = mode.loadImage(f"bgs/border.png")
        mode.backButt = ImageButton('back',x,mode.height-100,Image.open(f'splashButts/back.png'),Image.open(f'splashButts/backHover.png'))
        
    def redrawAll(mode, canvas):
        canvas.create_text(mode.width//2,mode.height//2,text='hello, about info here',fill='blue')
        canvas.create_image(mode.width//2, mode.height//2,image=ImageTk.PhotoImage(mode.border))
        mode.backButt.draw(canvas)
    
    def mouseMoved(mode,event):
        mode.backButt.checkHover(event.x,event.y)
    
    def mousePressed(mode,event):
        if(mode.backButt.isClicked(event.x,event.y)):
            mode.app.setActiveMode(mode.app.splashScreenMode)

# #fe4a49 • #2ab7ca • #fed766 • #e6e6ea • #f4f4f8
class Studio(Mode):
    def appStarted(self):
        # cv things
        self.recording = False
        self.vid = None
        self.frame = None
        self.vidOut = None
        
        # graphics things
        self.timerDelay = 20
        self.enterText = False
        self.currBubble = ["",0,0,None,1.75] # message, x, y
        self.doubleClickStarted, self.timeAt = False,time.time()
        self.draggedClip = None
        self.editing = False
        self.maxClips = 4
        self.prevRegion = None
        self.textSize = 1.75
        self.labels = []
        self.small = tkFont.Font(family='Avenir',size=13)  

        # margins
        self.pBarLeft, self.gBarTop, self.sBarTop = 870, 480, 200
        self.headerMarg = 40
        self.oMarg, self.iMarg = 40, 10
        
        self.importGraphics()
        self.initializeStudio()
        
    def initializeStudio(self):
        l = self.pBarLeft + self.oMarg
        x,y = (self.pBarLeft+self.width)//2, self.headerMarg+self.iMarg 
        self.recordButt = ImageButton("record",x,y, Image.open(f'controls/snap.png'),Image.open(f'controls/snapHover.png'))
        self.commandButt = ImageButton("command",x,y+75, Image.open(f'controls/helpMenu.png'),Image.open(f'controls/helpMenuHover.png'))
        self.saveButt = ImageButton('save',x,y+150,Image.open(f'controls/save.png'),Image.open(f'controls/saveHover.png'))
        self.controls = [self.recordButt,self.commandButt,self.saveButt]

        self.bg = Image.open(f'frame.png')

        x,y = self.oMarg+self.pBarLeft//2, self.gBarTop + self.headerMarg 
        self.default = ImageButton("default",x,y, Image.open(f'buttons/default.png'),Image.open(f'buttons/defaultHover.png'))
        self.dotCartoon = ImageButton("dot",x,  y+50,Image.open(f'buttons/pointilist.png'),Image.open(f'buttons/pointilistHover.png'))
        #self.pastel = ImageButton("pastel filter",l,                 self.oMarg + 200+self.iMarg,200,50)
        self.sketch = ImageButton("sketch",x,     y+100, Image.open(f'buttons/sketch.png'),Image.open(f'buttons/sketchHover.png'))
        self.vignette = ImageButton("vignette",x,  y+150, Image.open(f'buttons/vignette.png'),Image.open(f'buttons/vignetteHover.png'))
        self.benday = ImageButton("benday",x, y+200, Image.open(f'buttons/halftone.png'),Image.open(f'buttons/halftoneHover.png'))
        self.buttons = [self.dotCartoon, self.sketch,self.vignette,self.benday,self.default]
        
        self.studioRegion = StudioRegion("studio",0,self.headerMarg,self.pBarLeft, self.gBarTop-self.headerMarg,1/4,self.maxClips)
        self.savedRegion = SavedRegion("saved",self.pBarLeft,self.sBarTop+self.headerMarg,self.width-self.pBarLeft, self.gBarTop-self.sBarTop,0.1,10)
        self.graphicsRegion = GraphicsRegion('graphics',0,self.gBarTop,self.pBarLeft,self.height-self.gBarTop,0.3,20,self.graphics)
        self.regions = [self.studioRegion, self.savedRegion,self.graphicsRegion]


    
    # bubbles: 211558518, https://depositphotos.com/211558518/stock-illustration-big-set-empty-speech-bubble.html
    # black/white(graphics/bw): Max Luczynski, https://www.behance.net/gallery/47977047/One-Hundred-hand-drawn-cartoon-and-comic-symbols
    # explosions(graphics/colors): Tartila 20799751, https://www.vectorstock.com/royalty-free-vector/exclamation-texting-comic-signs-on-speech-bubbles-vector-20799751
    def importGraphics(self):
        self.graphics = []
        removeTempFiles('comics')
        for f in os.listdir('comics/bubbles'):
            img = self.loadImage(f"comics/bubbles/{f}")
            graphicAsClip = Clip(img,img,0,0,img.size[0],img.size[1],typ='bubble')
            self.graphics.append(graphicAsClip)
        for f in os.listdir('comics/colors'):
            img = self.loadImage(f"comics/colors/{f}")
            graphicAsClip = Clip(img,img,0,0,img.size[0],img.size[1],typ='graphic')
            self.graphics.append(graphicAsClip)
        for f in os.listdir('comics/bw'):
            img = self.loadImage(f"comics/bw/{f}")
            graphicAsClip = Clip(img,img,0,0,img.size[0],img.size[1],typ='graphic')
            self.graphics.append(graphicAsClip)
        removeTempFiles('labels')
        img = Image.open(f'labelsButts/editor.png')
        self.editorLabel = ('editor',self.oMarg+img.size[0]//2,self.oMarg+img.size[1]//2,img)
        img = Image.open(f'labelsButts/saved.png')
        self.savedLabel = ('saved',self.oMarg+self.pBarLeft+img.size[0]//2,self.oMarg+self.sBarTop+img.size[1]//2,img)
        img = Image.open(f'labelsButts/studio.png')
        self.studioLabel = ('studio',self.oMarg+img.size[0]//2,self.oMarg+img.size[1]//2,img)
        img = Image.open(f'labelsButts/effects.png')
        self.effectsLabel = ('effects',self.oMarg+img.size[0]//2,self.oMarg+self.gBarTop+img.size[1]//2,img)
        img = Image.open(f'labelsButts/edittext.png')
        self.editLabel = ('text',self.oMarg+self.pBarLeft+img.size[0]//2,self.oMarg+self.gBarTop+img.size[1]//2,img)
        self.labels = [self.editorLabel, self.savedLabel, self.studioLabel, self.effectsLabel, self.editLabel]

    def toggleRecordButton(self):
        self.recording = not self.recording
        if(self.recording):
            #self.recordButt.label = "click to stop recording"
            self.setupWebcam()
        else:
            #self.recordButt.label = "click to start recording"
            self.vid.release()
            cv2.destroyAllWindows()

    def checkBubbleClicked(self,region,event):
        if(type(region) == EditorRegion):
            coords = region.bubbleClicked(event.x,event.y)
            if(coords != None):
                self.enterText = True
                self.currBubble = ['',coords[0],coords[1],coords[2],coords[3]]
                return True
        return False

    def mouseMoved(self,event):
        if(self.editing):
            for butt in self.buttons:
                if(butt.checkHover(event.x,event.y)): return
        for butt in self.controls:
            if(butt.checkHover(event.x,event.y)): return

    def mousePressed(self,event):
        if(self.recordButt.isClicked(event.x,event.y)):
            self.toggleRecordButton()
            return
        elif(not self.editing and self.saveButt.isClicked(event.x,event.y)):
            self.studioRegion.finish() # implement title entry
        elif(self.commandButt.isClicked(event.x,event.y)):
            print('command butt clicked') # toggle overlay with command strip(maybe to the left?)
        if(self.editing):
            for butt in self.buttons:
                if(butt.isClicked(event.x,event.y)):
                    self.currEditorRegion.applyFilter(butt.label)
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
        if(not self.editing and event.key == 'k'):
            self.studioRegion.finish()
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
            if(event.key == 'Enter' or event.key == 'Escape'):
                self.enterText = False
                self.currEditorRegion.finishedBubble = True
                self.currEditorRegion.textUpdated = None
            elif(event.key == 'Up'):
                self.textSize += .25
                self.currEditorRegion.updateTextSize(self.textSize)
                self.currEditorRegion.textUpdated = ('',0,0,None,self.textSize)
            elif(event.key == 'Down'):
                self.textSize -= .25
                self.currEditorRegion.updateTextSize(self.textSize)
                self.currEditorRegion.textUpdated = ('',0,0,None,self.textSize)
            else:
                newChar = event.key
                mess,x,y,graphic,size = tuple(self.currBubble) # currently, abc
                if(newChar == 'Delete' and len(mess) > 0):
                    mess = mess[:len(mess)-1]
                else:
                    if(newChar == 'Space'):
                        newChar = ' '
                    if(len(mess)%(21-self.textSize*4) == 0):
                        if(len(mess) != 0 and mess[-1] != ' ' and newChar != ' '):
                            mess += '-`'
                        else:
                            mess += '`'
                    mess += newChar
                self.currBubble = mess,x,y,graphic,size
                self.currEditorRegion.insertBubbleText(mess,x,y,graphic,self.textSize)
                
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
        
    #def drawPanel(self,img):
        # create panel of buttons, drawn later
    #    self.filterB = cvButton("pink filter",0,0,self.w//5,self.h//5)
    #    self.filterB.draw(img)

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
            self.processVid()
        
        ret, self.frame = self.vid.read()
        self.imgCartoon = cart(self.frame) #AHHHH
        if(self.vidOut != None):
            self.writeToVid()
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
        canvas.create_line(self.pBarLeft,0,self.pBarLeft,self.height,fill='black',width=10)
        canvas.create_line(0,self.gBarTop,self.width,self.gBarTop,fill='black',width=10)
        for butt in self.buttons:
            butt.draw(canvas)
        for butt in self.controls:
            butt.draw(canvas)

    def drawFrame(self,canvas):
        canvas.create_image(self.width//2, self.height//2,image=ImageTk.PhotoImage(self.bg))

    def drawLabels(self,canvas):
        for label,x,y,img in self.labels:
            if(not(label == self.editorLabel[0] and not self.editing) and not(label == self.studioLabel[0] and self.editing)):
                canvas.create_image(x, y,image=ImageTk.PhotoImage(img))

    def drawTextBox(self,canvas):
        x, y = self.pBarLeft + self.oMarg+2*self.iMarg, self.gBarTop + self.oMarg + 95
        if(self.enterText):
            canvas.create_text(x,y,fill='gray',text=f'TEXT SIZE: {int(self.textSize*12)}\nPress up/down arrows to change.',font=self.small,anchor='nw')
        else:
            canvas.create_text(x,y,fill='gray',text=f'when you write in text bubbles, \nyour text will appear here!',font=self.small,anchor='nw')
            
    def redrawAll(self, canvas):
        self.drawFrame(canvas)
        self.drawControlPanel(canvas)
        self.drawBoard(canvas)
        self.drawLabels(canvas)
        self.drawTextBox(canvas)
    
class MyModalApp(ModalApp):
    def appStarted(app):
        app.splashScreenMode = SplashScreenMode()
        app.gameMode = Studio()
        app.helpMode = HelpMode()
        app.aboutMode = AboutMode()
        app.setActiveMode(app.splashScreenMode)
        app.timerDelay = 10

def playGame():
    MyModalApp(width = 1200, height = 800)

def main():
    playGame()

if __name__ == '__main__':
    main()
