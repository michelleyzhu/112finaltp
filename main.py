import math, copy, random, time, string, os
import cv2 as cv2
import numpy as np
#from tkinter import *
from cmu_112_graphics import *
import tkinter.font as tkFont
from cvhelpers import *
from fakecv import *
from widgets import *
from modes import *

##### CITATION: see importGraphics() for citation of images #####
bubbleRatio = 3.5

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
        self.bold = tkFont.Font(family='Avenir',size=14,weight='bold')  
        self.large = tkFont.Font(family='Avenir',size=18,weight='bold')  
        self.finishing = False
        self.title = ''
        self.congratsTimer,self.congrats = 0, False
        # margins
        self.pBarLeft, self.gBarTop, self.sBarTop = 870, 480, 200
        self.headerMarg = 40
        self.oMarg, self.iMarg = 40, 10
        
        self.importGraphics()
        self.initializeStudio()
        
    def initializeStudio(self):
        l = self.pBarLeft + self.oMarg
        x,y = (self.pBarLeft+self.width)//2, self.headerMarg+self.iMarg 
        self.recordButt = ImageButton("record",x,y, Image.open(f'graphics/controls/snap.png'),Image.open(f'graphics/controls/snapHover.png'))
        self.commandButt = ImageButton("command",x,y+75, Image.open(f'graphics/controls/helpMenu.png'),Image.open(f'graphics/controls/helpMenuHover.png'))
        self.saveButt = ImageButton('save',x,y+150,Image.open(f'graphics/controls/save.png'),Image.open(f'graphics/controls/saveHover.png'))
        self.controls = [self.recordButt,self.commandButt,self.saveButt]

        self.bg = Image.open(f'graphics/backgrounds/frame.png')

        x,y = 770, self.gBarTop + self.headerMarg +15
        self.default = ImageButton("default",x,y, Image.open(f'graphics/filters/default.png'),Image.open(f'graphics/filters/defaultHover.png'))
        self.dotCartoon = ImageButton("dot",x,  y+50,Image.open(f'graphics/filters/pointilist.png'),Image.open(f'graphics/filters/pointilistHover.png'))
        #self.pastel = ImageButton("pastel filter",l,                 self.oMarg + 200+self.iMarg,200,50)
        self.sketch = ImageButton("sketch",x,     y+100, Image.open(f'graphics/filters/sketch.png'),Image.open(f'graphics/filters/sketchHover.png'))
        self.vignette = ImageButton("vignette",x,  y+150, Image.open(f'graphics/filters/vignette.png'),Image.open(f'graphics/filters/vignetteHover.png'))
        self.benday = ImageButton("benday",x, y+200, Image.open(f'graphics/filters/halftone.png'),Image.open(f'graphics/filters/halftoneHover.png'))
        self.buttons = [self.dotCartoon, self.sketch,self.vignette,self.benday,self.default]
        

        img = Image.open(f'graphics/labels/sounds.png')
        self.soundsButton = ImageButton('sounds',self.oMarg+170+img.size[0]//2,self.oMarg+self.gBarTop+img.size[1]//2,img,Image.open(f'graphics/labels/soundsHover.png'),swapped=True)
        self.bwButton = ImageButton('bw',self.oMarg+320+2*img.size[0]//2,self.oMarg+self.gBarTop+img.size[1]//2,Image.open(f'graphics/labels/graphics.png'),Image.open(f'graphics/labels/graphicsHover.png'),swapped=False)
        self.bubblesButton = ImageButton('speech',self.oMarg+470+3*img.size[0]//2,self.oMarg+self.gBarTop+img.size[1]//2,Image.open(f'graphics/labels/bubbles.png'),Image.open(f'graphics/labels/bubblesHover.png'),swapped=False)
        
        self.currGraphicButton = self.soundsButton
        self.graphicButtons = [self.soundsButton,self.bwButton,self.bubblesButton]

        self.studioRegion = StudioRegion("studio",0,self.headerMarg,self.pBarLeft, self.gBarTop-self.headerMarg,1/4,self.maxClips)
        self.savedRegion = SavedRegion("saved",self.pBarLeft+15,self.sBarTop+self.headerMarg+30,self.width-self.pBarLeft, self.gBarTop-self.sBarTop,0.08,10)
        
        x,y,w,h = 0,self.gBarTop+20,self.pBarLeft-200,self.height-self.gBarTop-20
        self.bubbleRegion = GraphicsRegion('speech',x,y,w,h,0.3,20,self.bubbles)
        self.soundsRegion = GraphicsRegion('sounds',x,y,w,h,0.3,20,self.sounds)
        self.bwRegion = GraphicsRegion('bw',x,y,w,h,0.3,20,self.bw)

        self.selectGraphicRegion(self.soundsRegion)
        self.graphicsRegions = [self.bubbleRegion,self.soundsRegion,self.bwRegion]

        self.regions = [self.studioRegion, self.savedRegion,self.bubbleRegion,self.soundsRegion,self.bwRegion]

    def selectGraphicRegion(self,reg):
        if(reg == self.bubbleRegion):
            self.bubbleRegion.active = True
            self.soundsRegion.active = False
            self.bwRegion.active = False
        elif(reg == self.soundsRegion):
            self.bubbleRegion.active = False
            self.soundsRegion.active = True
            self.bwRegion.active = False
        elif(reg == self.bwRegion):
            self.bubbleRegion.active = False
            self.soundsRegion.active = False
            self.bwRegion.active = True


    # more bubles: https://www.vectorstock.com/royalty-free-vector/empty-monochrome-speech-comic-text-bubbles-vector-13248595
    # bubbles: 211558518, https://depositphotos.com/211558518/stock-illustration-big-set-empty-speech-bubble.html
    # black/white(graphics/bw): Max Luczynski, https://www.behance.net/gallery/47977047/One-Hundred-hand-drawn-cartoon-and-comic-symbols
    # explosions(graphics/colors): Tartila 20799751, https://www.vectorstock.com/royalty-free-vector/exclamation-texting-comic-signs-on-speech-bubbles-vector-20799751
    def importGraphics(self):
        self.bubbles, self.bw, self.sounds = [],[],[]
        for f in os.listdir('graphics/comics/bubbles'): # BUBBLES
            img = self.loadImage(f"graphics/comics/bubbles/{f}")
            if(f[0] == 'r'):
                cv = pilToCV(img)
                cv = mirrorImage(cv)
                img = cvToPIL(cv)
                direct = 'right'
            elif(f[0] == 'l'):
                direct = 'left'
            else:
                direct = 'norm'
            graphicAsClip = Clip(img,img,0,0,img.size[0],img.size[1],typ='bubble',direct=direct)
            self.bubbles.append(graphicAsClip)
        for f in os.listdir('graphics/comics/colors'): # COLORS
            img = self.loadImage(f"graphics/comics/colors/{f}")
            graphicAsClip = Clip(img,img,0,0,img.size[0],img.size[1],typ='graphic')
            self.sounds.append(graphicAsClip)
        for f in os.listdir('graphics/comics/bw'): # BW
            img = self.loadImage(f"graphics/comics/bw/{f}")
            graphicAsClip = Clip(img,img,0,0,img.size[0],img.size[1],typ='graphic')
            self.bw.append(graphicAsClip)
        img = Image.open(f'graphics/labels/editor.png')
        self.editorLabel = ('editor',self.oMarg+img.size[0]//2,self.oMarg+img.size[1]//2,img)
        img = Image.open(f'graphics/labels/saved.png')
        self.savedLabel = ('saved',self.oMarg+self.pBarLeft+img.size[0]//2,self.sBarTop+self.headerMarg+60,img)
        img = Image.open(f'graphics/labels/studio.png')
        self.studioLabel = ('studio',self.oMarg+img.size[0]//2,self.oMarg+img.size[1]//2,img)
        img = Image.open(f'graphics/labels/effects.png')
        self.effectsLabel = ('effects',self.oMarg+img.size[0]//2,self.oMarg+self.gBarTop+img.size[1]//2,img)
        img = Image.open(f'graphics/labels/edittext.png')
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
            self.finishing = True
            self.enterText = True
            return
        elif(self.commandButt.isClicked(event.x,event.y)):
            print('command butt clicked') # toggle overlay with command strip(maybe to the left?)
            return
        
        for butt in self.graphicButtons:
            if(butt.isClicked(event.x,event.y) and butt.label != self.currGraphicButton.label):# and self.graphicsRegionActive(butt.label)):
                self.selectGraphicRegion(self.regionWithName(butt.label))
                butt.swapHover()
                self.currGraphicButton.swapHover()
                self.currGraphicButton = butt
        
        if(self.editing):
            for butt in self.buttons:
                if(butt.isClicked(event.x,event.y)):
                    self.currEditorRegion.applyFilter(butt.label)
                    return
        for region in self.regions:
            if(not region.active): continue # if inactive, don't respond to dragging
            for i in range(len(region.drawables)):
                drawable = region.drawables[i]
                if drawable.typ == 'bubble' and self.checkBubbleClicked(region,event): return # CHECKS BUBBLE CLICKS
                if(region.canMove(i) and drawable.isClicked(event.x,event.y)):
                    self.dragX, self.dragY, self.draggedClip = event.x,event.y,drawable
                    self.prevRegion = region
                    if(not self.isGraphicsRegion(region.name)): # if graphics, double clicking shouldn't do anything
                        if(self.doubleClickStarted):
                            if(time.time() - self.timeAt <= 1):
                                self.openEditor(drawable)
                            self.doubleClickStarted = False
                        else:
                            self.doubleClickStarted = True
                            self.timeAt = time.time()
                    return
    
    def regionWithName(self,regName):
        for reg in self.graphicsRegions:
            if(reg.name == regName):
                return reg

    def graphicsRegionActive(self,regName):
        for reg in self.graphicsRegions:
            if(reg.name == regName):
                if(reg.active):
                    return True
                return False

    def isGraphicsRegion(self,regName):
        return regName == self.soundsRegion.name or regName == self.bwRegion.name or regName == self.bubbleRegion.name

    def openEditor(self,drawable):
        self.editing = True
        self.currEditorRegion = drawable.editor
        for region in self.regions:
            if(not self.isGraphicsRegion(region.name)):
                region.active = False
        drawable.editor.active = True
        
    def keyPressed(self,event):
        # can't escape when entering text
        if(self.finishing):
            if(event.key == 'Enter'):
                self.studioRegion.finish(self.title)
                self.finishing = False
                self.enterText = False
                self.congratsTimer,self.congrats = time.time(), True
            else:
                newChar = event.key
                if(newChar == 'Delete' and len(self.title) > 0):
                    self.title = self.title[:len(self.title)-1]
                else:
                    if(newChar == 'Space'):
                        newChar = ' '
                    self.title += newChar
        elif(self.editing and not self.enterText and event.key == "Escape"):
            self.editing = False
            for region in self.regions:
                i = 0
                for drawable in region.drawables:
                    if(type(region) != EditorRegion and drawable.editor == self.currEditorRegion and (type(region) == StudioRegion and region.shownDrawables[i])):
                        drawable.package(drawable)
                    i+=1
                if(type(region) != EditorRegion):
                    region.active = True
                self.selectGraphicRegion(self.regionWithName(self.currGraphicButton.label))
            self.currEditorRegion.active = False
        elif(self.enterText):
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
        self.vidOut.write(self.frame)

    def processVid(self):
        v = cv2.VideoCapture("vidOut.avi")
        self.faceVid = cv2.VideoWriter("faceVid.avi",cv2.VideoWriter_fourcc("M","J","P","G"),5,(self.w,self.h),True)
        faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
        self.totalFrames, self.faceFrames = 0,0
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
                mx,my,mw,mh = largestFace
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
                self.faceVid.write(frame)
                self.faceFrames += 1
            self.totalFrames += 1
        self.packageFrames()
    
    def packageFrames(self):
        faceIndices = sorted(random.sample([i for i in range(self.faceFrames)],min(self.faceFrames,4)))
        self.autoImages = []
        self.AutoFrames = []
        f = cv2.VideoCapture('faceVid.avi')
        v = cv2.VideoCapture('vidOut.avi')
        for i in range(self.faceFrames):
            ret, frame = f.read()
            if(i == faceIndices[0]):
                print(f'reading frame {i+1}')
                self.autoImages.append(frame)
                imgClip = self.processFrame(frame)
                self.AutoFrames.append(imgClip)
                if(len(faceIndices) > 1):
                    faceIndices = faceIndices[1:]
                else:
                    break
        while(len(self.AutoFrames) < min(4,self.totalFrames)):
            ret, frame = v.read()
            if(not np.array_equal(frame, self.autoImages)):
                imgClip = self.processFrame(frame,False)
                self.AutoFrames.append(imgClip) # The appended should be clip(drawable) objects
        for clip in self.AutoFrames:
            self.studioRegion.addDrawable(clip,self.savedRegion)
            self.regions.append(clip.editor)

    def processFrame(self,cvImg,face=True):
        faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
        mouthCascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")
        
        if(face):
            faces = faceCascade.detectMultiScale(cvImg)
            largestFace, area = None,0
            for x,y,w,h in faces:
                if(largestFace == None or w*h > area):
                    largestFace, area = (x,y,w,h), w*h
            if(largestFace != None and area > 40000):
                fX,fY,fW,fH = largestFace
                cv2.rectangle(cvImg,(fX,fY),(fX+fW,fY+fH),(0,255,0),10) # face is thicker
                mouths = mouthCascade.detectMultiScale(cvImg[fY+3*fH//4:fY+fH,fX:fX+fW])
                largestMouth, area = None, 0
                for mx,my,mw,mh in mouths:
                    if(largestMouth == None or mw*mh > area):
                        largestMouth, area = (mx,my,mw,mh), mw*mh
                if(largestMouth != None):
                    mX,mY,mW,mH = largestMouth
                    cv2.rectangle(cvImg,(fX+mX,fY+3*fH//4+mY),(fX+mX+mW,fY+3*fH//4+mY+mH),(0,255,0),5)
                    leftX, rightX, pointY = fX,fX+fW,fY+3*fH//4+mY-20
                    
                    currIm = cvToPIL(cvImg)
                    w, h = currIm.size[0], currIm.size[1]
                    newClip = Clip(currIm,cvImg,0,0,w,h,editor=[])
        
                    if(not self.placeSides(newClip,leftX,rightX,pointY)):
                        print('failed to add to the sides')
                        self.placePlain(newClip)
                    newClip.editor.applyFilter('default')
                    return newClip
        currIm = cvToPIL(cvImg)
        w, h = currIm.size[0], currIm.size[1]
        newClip = Clip(currIm,cvImg,0,0,w,h,editor=[])
        self.placePlain(newClip)
        newClip.editor.applyFilter('default')
        return newClip

    def placeSides(self,clip,leftX,rightX,pointY):
        bubbs = self.bubbles.copy()
        random.shuffle(bubbs)
        for bubble in bubbs: # bubble is a clip object graphic
            if(bubble.direct == 'left'):
                origW,origH = bubble.w, bubble.h
                w,h = bubbleRatio*origW/clip.scale, bubbleRatio*origH/clip.scale #origW,origH #
                bubble.x, bubble.y = int(leftX-w), int(pointY-h)
                #print(f'w,h: {origW,origH}, and scaled: {w,h}')
                print(f'bubble x/y on the right: {bubble.x,bubble.y}')
                
                clip.editor.addDrawable(bubble,clip.editor)
                
                scaledGraphic = bubble.img.resize((int(bubble.img.size[0]*bubbleRatio),int(bubble.img.size[1]*bubbleRatio)))
                mask = pilToCV(scaledGraphic)
                x1 = int((bubble.x-clip.editor.x0)/clip.editor.scale*1 - clip.editor.oMarg)
                y1 = int((bubble.y-clip.editor.y0)/clip.editor.scale - clip.editor.headerMarg)
                
                print(mask.shape)
                print(f'top left: {x1},{y1}, bot right: {x1+mask.shape[1],y1+mask.shape[0]}, limit: {clip.origImg.shape[1],clip.origImg.shape[0]}')
                if(0 <= x1 and x1+mask.shape[1] <= clip.origImg.shape[1] and 0 <= y1 and y1+mask.shape[0] <= clip.origImg.shape[0]):
                    clip.editor.updateGraphics()
                    print('shouldve added to left succesfully!')
                    return True
                else:
                    if(bubble in clip.editor.drawables[1:]):
                        clip.editor.removeDrawable(bubble)
                    print(f'failed to add {bubble.name}to the left')
            elif(bubble.direct == 'right'):
                origW,origH = bubble.w, bubble.h
                w,h = bubbleRatio*origW/clip.scale, bubbleRatio*origH/clip.scale #origW,origH #
                bubble.x, bubble.y = int(rightX), int(pointY-h)
                print(f'bubble x/y on the right: {bubble.x,bubble.y}')
                
                clip.editor.addDrawable(bubble,clip.editor)
                
                scaledGraphic = bubble.img.resize((int(bubble.img.size[0]*bubbleRatio),int(bubble.img.size[1]*bubbleRatio)))
                mask = pilToCV(scaledGraphic)
                x1 = int((bubble.x-clip.editor.x0)/clip.editor.scale*1 - clip.editor.oMarg)
                y1 = int((bubble.y-clip.editor.y0)/clip.editor.scale - clip.editor.headerMarg)
                
                print(mask.shape)
                if(5 <= x1 and x1+mask.shape[1] <= clip.origImg.shape[1]-5 and 5 <= y1 and y1+mask.shape[0] <= clip.origImg.shape[0]-5):
                    clip.editor.updateGraphics()
                    print('shouldve added to right succesfully!')
                    return True
                else:
                    if(bubble in clip.editor.drawables[1:]):
                        clip.editor.removeDrawable(bubble)
                    print(f'failed to add {bubble.name}to the right')
        print('failed to add to both sides')
        return False

    def placePlain(self,clip):
        bubbs = self.bubbles.copy()
        random.shuffle(bubbs)
        for bubble in bubbs: # bubble is a clip object graphic
            if(bubble.direct == 'norm'):
                #origW,origH = bubble.w, bubble.h
                #w,h = origW/clip.scale, origH/clip.scale
                
                bubble.x, bubble.y = 50, 50
                print('trying the norm:')
                
                clip.editor.addDrawable(bubble,clip.editor)
                scaledGraphic = bubble.img.resize((int(bubble.img.size[0]*bubbleRatio),int(bubble.img.size[1]*bubbleRatio)))
                mask = pilToCV(scaledGraphic)
                x1 = int((bubble.x-clip.editor.x0)/clip.editor.scale*1 - clip.editor.oMarg)
                y1 = int((bubble.y-clip.editor.y0)/clip.editor.scale - clip.editor.headerMarg)
                
                print(mask.shape)
                if(5 <= x1 and x1+mask.shape[1] <= clip.origImg.shape[1]-5 and 5 <= y1 and y1+mask.shape[0] <= clip.origImg.shape[0]-5):
                    clip.editor.updateGraphics()
                    return True
                else:
                    if(bubble in clip.editor.drawables[1:]):
                        clip.editor.removeDrawable(bubble)
                    print(f'failed to add {bubble.name}to the normal')

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
        if(self.congrats and time.time()-self.congratsTimer >= 3):
            self.congrats = False
            self.title = ''
    
    def drawBoard(self,canvas):
        if(self.editing):
            self.currEditorRegion.draw(canvas,False)
            for region in self.graphicsRegions:
                if(region.active):
                    region.draw(canvas,False)
        else:
            for region in self.regions:
                if(region.active):
                    region.draw(canvas)
            
    def drawControlPanel(self,canvas):
        #canvas.create_line(self.pBarLeft,0,self.pBarLeft,self.height,fill='black',width=10)
        #canvas.create_line(0,self.gBarTop,self.width,self.gBarTop,fill='black',width=10)
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
        if(self.finishing):
            canvas.create_text(x,y+20,fill='gray',text='enter the title of your\ncomic and press ENTER to save:',font=self.bold,anchor='nw')
            canvas.create_text(x,y+60,fill='gray',text=self.title,font=self.small,anchor='nw')
        elif(self.enterText):
            canvas.create_text(x,y,fill='gray',text=f'TEXT SIZE: {int(self.textSize*12)}\nPress up/down arrows to change.',font=self.small,anchor='nw')
        else:
            canvas.create_text(x,y,fill='gray',text=f'when you write in text bubbles, \nyour text will appear here!',font=self.small,anchor='nw')
            
    def drawGraphicsButtons(self,canvas):
        for butt in self.graphicButtons:
            butt.draw(canvas)

    def redrawAll(self, canvas):
        self.drawFrame(canvas)
        self.drawControlPanel(canvas)
        self.drawGraphicsButtons(canvas)
        self.drawBoard(canvas)
        self.drawLabels(canvas)
        self.drawTextBox(canvas)
        if(self.congrats):
            canvas.create_text(200,self.height//2,fill='gray',text=f'nice! \'{self.title}\' saved to finalProduct.png, go take a look!',font=self.large,anchor='nw')
    
class MyModalApp(ModalApp):
    def appStarted(app):
        removeTempFiles('graphics')
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
