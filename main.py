import math, copy, random, time, string, os
import cv2 as cv2
import numpy as np
from cmu_112_graphics import *
import tkinter.font as tkFont
from cvInterface import *
from widgets import *
from modes import *

############################################################################
# main.py:
# This controls my main Studio mode app, and includes the primary functions
# of my MVC control as well as video-processing methods for clip auto-
# generation.
############################################################################

faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
mouthCascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")
    
##### CITATION: see importGraphics() for citation of images #####

class Studio(Mode):
    def appStarted(self):
        # cv objects
        self.vid = None
        self.frame = None
        self.vidOut = None
        # graphics things
        self.timerDelay = 10
        self.currBubble = ["",0,0,None,1.75] # message, x, y
        self.draggedClip = None
        self.prevRegion = None
        self.currEditorRegion = None
        self.graphicsRegions = None
        self.imgCartoon = None
        # flags
        self.recording = False
        self.editing = False
        self.doubleClickStarted, self.timeAt = False,time.time()
        self.congratsTimer,self.congrats = 0, False
        self.finishing = False
        # text entry
        self.enterText = False
        self.textSize = 1.75
        self.small = tkFont.Font(family='Avenir',size=13)  
        self.bold = tkFont.Font(family='Avenir',size=14,weight='bold')  
        self.large = tkFont.Font(family='Avenir',size=18,weight='bold')  
        self.title = ''
        # margins
        self.pBarLeft, self.gBarTop, self.sBarTop = 870, 480, 200
        self.headerMarg = 40
        self.oMarg, self.iMarg = 40, 10
        self.bubbleRatio = 2.5
        # More initialization - categorized!
        self.importGraphics()
        self.initializeStudio()
        self.defineComments()
    
    def defineComments(self):
        self.sequences = {
            'doubt':['I may not be`great at art...','or anything`in general...','what was my`point again?','ah, code`is good'],
            'dark':['Why is it so`dark in here??','ah, I know...',"it's the absence`of all intelligence.",'#hardcodedburn'],
            'empty':["OMG look away","I'm camera shy","and I won't`let the FBI see`my face",'cover']
        }
        self.emotes = {
            'anger':['The comic sans`font sparks`anger in me.',"Well, I'm angry`it's not in`this app!","Michelle SHOULD'VE`put it in...","but she was`TIRED, I guess."],
            'disgust':['The comic sans`font is SO`gross.',"I'm disgusted`that it's not`in this app!","Michelle is so`lame for not`pulling it off",'I guess she`was too "tired"'],
            'fear':["The comic sans`font scares me.","Well, I'm scared`about why it's`not in this app!","shouldn't michelle`have put`it in??","what prevented`her from pulling`that off?"],
            'happy':["The comic sans`font brings me`all of my joy.","Well, I'm okay`with not having`it in this app!","wouldn't it`have been cool`to have that?","it's fine that`she couldn't`do it."],
            'sad':["The comic sans`font depresses`me.","Well, I'm sad`it's not in`this app!","It would've been`nice to have...","she was probably`too tired and`dumb. :(("],
            'neutral':["The comic sans`font is cool.","Well, I wish`it was in`this app.","It would be`a nice feature,","oh well, she can't`do it all."],
            'surprise':["The comic sans`font always`startles me.","I was shocked`to see it's not`in this app!","It would've been`SO cool to have,","and I can't`believe she didn't`pull that off!"],
            'none':["This nerd likes`comic sans font.","they want it to be in the app.","comic sans`would've been`cool...","the nerd is sad."]
        }
        
    def initializeStudio(self):
        self.backgroundImage = Image.open(f'graphics/backgrounds/frame.png')
        self.blueBorder = Image.open(f'graphics/labels/blueBorder.png')

        # Primary Control Buttons
        x,y = (self.pBarLeft+self.width)//2, self.headerMarg+self.iMarg 
        self.recordButt = ImageButton("record",x,y, Image.open(f'graphics/controls/snap.png'),Image.open(f'graphics/controls/snapHover.png'))
        self.commandButt = ImageButton("command",x,y+75, Image.open(f'graphics/controls/helpMenu.png'),Image.open(f'graphics/controls/helpMenuHover.png'))
        self.saveButt = ImageButton('save',x,y+150,Image.open(f'graphics/controls/save.png'),Image.open(f'graphics/controls/saveHover.png'))
        self.trashButt = ImageButton('trash',self.pBarLeft-75,self.gBarTop-130,Image.open(f'graphics/controls/close.png'),Image.open(f'graphics/controls/open.png'))
        self.backButt = ImageButton('back',self.pBarLeft-75,self.gBarTop-50,Image.open(f'graphics/controls/back.png'),Image.open(f'graphics/controls/backHover.png'))
        self.undoButt = ImageButton('undo',self.pBarLeft-75,self.gBarTop-200,Image.open(f'graphics/controls/undo.png'),Image.open(f'graphics/controls/undoHover.png'))
        
        self.controls = [self.recordButt,self.commandButt,self.saveButt,self.trashButt,self.backButt,self.undoButt]
        
        # Filter Buttons
        x,y = 770, self.gBarTop + self.headerMarg +15
        self.default = ImageButton("default",x,y, Image.open(f'graphics/filters/default.png'),Image.open(f'graphics/filters/defaultHover.png'))
        self.dotCartoon = ImageButton("dot",x,  y+50,Image.open(f'graphics/filters/pointilist.png'),Image.open(f'graphics/filters/pointilistHover.png'))
        self.sketch = ImageButton("sketch",x,     y+100, Image.open(f'graphics/filters/sketch.png'),Image.open(f'graphics/filters/sketchHover.png'))
        self.vignette = ImageButton("vignette",x,  y+150, Image.open(f'graphics/filters/vignette.png'),Image.open(f'graphics/filters/vignetteHover.png'))
        self.benday = ImageButton("benday",x, y+200, Image.open(f'graphics/filters/halftone.png'),Image.open(f'graphics/filters/halftoneHover.png'))
        self.filterButtons = [self.dotCartoon, self.sketch,self.vignette,self.benday,self.default]
        
        # Graphics Tabs Buttons
        img = Image.open(f'graphics/labels/sounds.png')
        self.soundsButton = ImageButton('sounds',self.oMarg+170+img.size[0]//2,self.oMarg+self.gBarTop+img.size[1]//2,img,Image.open(f'graphics/labels/soundsHover.png'),swapped=True)
        self.bwButton = ImageButton('bw',self.oMarg+320+2*img.size[0]//2,self.oMarg+self.gBarTop+img.size[1]//2,Image.open(f'graphics/labels/graphics.png'),Image.open(f'graphics/labels/graphicsHover.png'),swapped=False)
        self.bubblesButton = ImageButton('speech',self.oMarg+470+3*img.size[0]//2,self.oMarg+self.gBarTop+img.size[1]//2,Image.open(f'graphics/labels/bubbles.png'),Image.open(f'graphics/labels/bubblesHover.png'),swapped=False)
        #if bgs: +250,+330
        #self.bgsButton = ImageButton('bgs',self.oMarg+460+3*img.size[0]//2,self.oMarg+self.gBarTop+img.size[1]//2-4,Image.open(f'graphics/labels/backgrounds.png'),Image.open(f'graphics/labels/backgroundsHover.png'),swapped=False)
        
        self.currGraphicButton = self.soundsButton
        self.graphicButtons = [self.soundsButton,self.bwButton,self.bubblesButton] #,self.bgsButton

        # Regions
        self.studioRegion = StudioRegion("studio",0,self.headerMarg,self.pBarLeft, self.gBarTop-self.headerMarg,1/4,4)
        self.savedRegion = SavedRegion("saved",self.pBarLeft+15,self.sBarTop+self.headerMarg+30,self.width-self.pBarLeft, self.gBarTop-self.sBarTop,0.08,10)
        
        x,y,w,h = 0,self.gBarTop+20,self.pBarLeft-200,self.height-self.gBarTop-20
        self.bubbleRegion = GraphicsRegion('speech',x,y,w,h,0.2,20,self.bubbles)
        self.soundsRegion = GraphicsRegion('sounds',x,y,w,h,0.3,20,self.sounds)
        self.bwRegion = GraphicsRegion('bw',x,y,w,h,0.3,20,self.bw)
        #self.bgsRegion = GraphicsRegion('bgs',x,y,w,h,0.05,20,self.bgs)

        self.graphicsRegions = [self.bubbleRegion,self.soundsRegion,self.bwRegion] #,self.bgsRegion
        self.selectGraphicRegion(self.soundsRegion)
        # and then, all regions
        self.regions = [self.studioRegion, self.savedRegion,self.bubbleRegion,self.soundsRegion,self.bwRegion] #,self.bgsRegion

    # black/white(graphics/bw): Max Luczynski, https://www.behance.net/gallery/47977047/One-Hundred-hand-drawn-cartoon-and-comic-symbols
    # explosions(graphics/colors): Tartila 20799751, https://www.vectorstock.com/royalty-free-vector/exclamation-texting-comic-signs-on-speech-bubbles-vector-20799751
    # aesthetics(backgrounds, buttons, text bubbles): Katie Shaw(CMU 2024)
    def importGraphics(self):
        self.bubbles, self.bw, self.sounds = [],[],[] #, self.bgs,[]
        for f in os.listdir('graphics/comics/bubbles'): # BUBBLES
            img = self.loadImage(f"graphics/comics/bubbles/{f}")
            if(f[0] == 'r'):
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
        for f in os.listdir('graphics/comics/narrations'): # NARRATIONS
            img = self.loadImage(f"graphics/comics/narrations/{f}")
            graphicAsClip = Clip(img,img,0,0,img.size[0],img.size[1],typ='narration')
            self.bubbles.append(graphicAsClip)
        '''for f in os.listdir('graphics/comics/bgs'): # NARRATIONS
            img = self.loadImage(f"graphics/comics/bgs/{f}")
            graphicAsClip = Clip(img,img,0,0,img.size[0],img.size[1],typ='graphic') # ACTUALLY BACKGROUND!!!!!
            self.bgs.append(graphicAsClip)'''
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
    
    def timerFired(self):
        if(self.recording):
            self.record()
        if(self.congrats and time.time()-self.congratsTimer >= 3):
            self.congrats = False
            self.title = ''
           
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

    def mouseMoved(self,event):
        for butt in self.filterButtons:
            if(butt.checkHover(event.x,event.y)): return
        for butt in self.controls:
            if(butt.checkHover(event.x,event.y)): return
        for butt in self.graphicButtons:
            if(butt.label != self.currGraphicButton.label and butt.checkHover(event.x,event.y)): return

    def mousePressed(self,event):
        if(self.recordButt.isClicked(event.x,event.y)):
            self.toggleRecordButton()
            return
        elif(not self.editing and self.saveButt.isClicked(event.x,event.y)):
            self.finishing = True
            self.enterText = True
            return
        elif(self.commandButt.isClicked(event.x,event.y)):
            self.app.setActiveMode(self.app.editorHelpMode)
            return
        elif(self.trashButt.isClicked(event.x,event.y)):
            self.studioRegion.clearAll()
            return
        elif(self.backButt.isClicked(event.x,event.y)):
            self.app.setActiveMode(self.app.splashScreenMode)
        elif(self.editing and not self.enterText and self.undoButt.isClicked(event.x,event.y)):
            self.currEditorRegion.undo()
        for butt in self.graphicButtons:
            if(butt.isClicked(event.x,event.y) and butt.label != self.currGraphicButton.label):# and self.graphicsRegionActive(butt.label)):
                self.selectGraphicRegion(self.regionWithName(butt.label))
                butt.swapHover()
                butt.hover = False
                self.currGraphicButton.swapHover()
                self.currGraphicButton = butt
        
        for butt in self.filterButtons:
            if(butt.isClicked(event.x,event.y)):
                if(self.editing):
                    self.currEditorRegion.applyFilter(butt.label)
                    return
                else:
                    for clip in self.studioRegion.drawables:
                        if(clip.editor != None):
                            clip.editor.applyFilter(butt.label)
                            clip.package()
                    return
        for region in self.regions:
            if(not region.active): continue # if inactive, don't respond to dragging
            for i in range(len(region.drawables)):
                drawable = region.drawables[i]
                if (drawable.typ == 'bubble' or drawable.typ == 'narration') and self.checkBubbleClicked(region,event): return # CHECKS BUBBLE CLICKS
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
                        region.addDrawable(drawable,self.savedRegion)
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
                    if(len(mess)%int(25-self.textSize*7) == 0): #25-self.textSize*7
                        if(len(mess) != 0 and mess[-1] != ' ' and newChar != ' '):
                            mess += '-`'
                        else:
                            mess += '`'
                    mess += newChar
                self.currBubble = mess,x,y,graphic,size
                self.currEditorRegion.insertBubbleText(mess,x,y,graphic,self.textSize)
    
    def selectGraphicRegion(self,reg):
        if(self.graphicsRegions != None):
            for region in self.graphicsRegions:
                region.active = False
                if(reg == region):
                    region.active = True

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
        for region in self.graphicsRegions:
            if(region.name == regName):
                return True
        return False

    def openEditor(self,drawable):
        self.editing = True
        self.currEditorRegion = drawable.editor
        for region in self.regions:
            if(not self.isGraphicsRegion(region.name)):
                region.active = False
        drawable.editor.active = True
        
    def checkBubbleClicked(self,region,event):
        if(type(region) == EditorRegion):
            coords = region.bubbleClicked(event.x,event.y)
            if(coords != None):
                self.enterText = True
                self.currBubble = ['',coords[0],coords[1],coords[2],coords[3]]
                return True
        return False

    ########################################################################
    ################## CV + Processing Methods #############################
    ########################################################################

    def toggleRecordButton(self):
        self.recording = not self.recording
        if(self.recording):
            self.setupWebcam()
        else:
            self.vid.release()
            cv2.destroyAllWindows()

    def setupWebcam(self):
        self.vid = cv2.VideoCapture(0)
        ret, self.frame = self.vid.read(0)
        self.windowFrac = 1/3
        self.w, self.h = self.frame.shape[1], self.frame.shape[0]
        self.minW, self.minH = int(self.w*self.windowFrac), int(self.h*self.windowFrac)
        cv2.namedWindow('commands')
        cv2.moveWindow('commands',100,100)
        cv2.imshow('commands',cv2.imread('graphics/webcam.png'))
        cv2.namedWindow('recording')
        #cv2.setMouseCallback('recording',self.cvMouseActed)
    
    def record(self):
        key = cv2.waitKey(1)
        if(key == ord('q')):
            self.recording = False
            self.vid.release()
            cv2.destroyAllWindows()
            return
        elif(key == ord('s')): # and type(self.imgCartoon) != NoneType
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
            return
        
        ret, self.frame = self.vid.read()
        self.resized = cv2.resize(self.frame,(self.w,self.h)) #self.w//2,self.h//2
        self.imgCartoon = cart(self.resized) #AHHHH
        self.imgCartoon = cv2.resize(self.imgCartoon,(self.minW,self.minH)) #AHHHH
        if(self.vidOut != None):
            self.vidOut.write(self.frame)
        cv2.imshow("recording",self.imgCartoon)
        
    def saveSnap(self):
        # the working/showing image is "img" and is a PIL Image object
        # the original stored image will be used to update the image, and is a CV-compatible nparray
        saved = cart(self.frame)
        cv2.imwrite("savedIm.jpg",saved)
        currIm = self.loadImage("savedIm.jpg")
        w, h = currIm.size[0], currIm.size[1]
        newClip = Clip(currIm,self.frame,0,0,w,h,editor=[])
        self.studioRegion.addDrawable(newClip,self.savedRegion)
        self.regions.append(newClip.editor)
    
    ########################################################################
    ################## Video Processing, Auto-Generation ###################
    ########################################################################

    def processVid(self):
        v = cv2.VideoCapture("vidOut.avi")
        self.faceVid = cv2.VideoWriter("faceVid.avi",cv2.VideoWriter_fourcc("M","J","P","G"),5,(self.w,self.h),True)
        #faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
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
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
                self.faceVid.write(frame)
                self.faceFrames += 1
            self.totalFrames += 1
        self.packageFrames()
    
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
    
    def packageFrames(self):
        if(self.faceFrames > 8):
            faceIndices = sorted(random.sample([i for i in range(self.faceFrames)[::2]],4))
        else:
            faceIndices = sorted(random.sample([i for i in range(self.faceFrames)],min(self.faceFrames,4)))
        self.autoImages = []
        self.AutoFrames = []
        self.describeFaces = []
        self.isDark = []
        self.emotions = []
        f = cv2.VideoCapture('faceVid.avi')
        v = cv2.VideoCapture('vidOut.avi')
        for i in range(self.faceFrames):
            ret, frame = f.read()
            if(i == faceIndices[0]):
                self.autoImages.append(frame)
                self.isDark.append(isDark(frame))
                self.emotions.append(getEmotion(frame)) # HEREHRHREHREHRHEHREHHRHEHREHRHEHREHRHEHRHEREHRERE
                imgClip = self.processFrame(frame)
                self.AutoFrames.append(imgClip)
                self.describeFaces.append(True)
                if(len(faceIndices) > 1):
                    faceIndices = faceIndices[1:]
                else:
                    break
        while(len(self.AutoFrames) < min(4,self.totalFrames)):
            ret, frame = v.read()
            if(not self.alreadyIncluded(frame)):
                imgClip = self.processFrame(frame,False)
                self.AutoFrames.append(imgClip) # The appended should be clip(drawable) objects
                self.describeFaces.append(False)
        self.commentate()

    def alreadyIncluded(self,frame):
        for included in self.autoImages:
            if(np.array_equal(frame,included)):
                return True
        return False

    def commentate(self):
        if(not self.any(self.describeFaces)): self.currMood = 'empty'
        elif(self.all(self.isDark)): self.currMood = 'dark'
        elif(self.emotions.count('none') < 3 and len(self.emotions) == 4):
            newSeq = []
            for i in range(len(self.emotions)):
                newSeq.append(self.emotes[self.emotions[i]][i])
            self.sequences['my own sequence'] = newSeq
            self.currMood = 'my own sequence'
        else: self.currMood = 'doubt'
        i = 0
        for clip in self.AutoFrames:
            editor = clip.editor
            if(len(editor.drawables)>1):
                bubble = editor.drawables[1]
                mess,x,y = self.sequences[self.currMood][i], bubble.x + bubble.w*self.bubbleRatio*editor.scale//2,bubble.y + bubble.h*self.bubbleRatio*editor.scale//2
                editor.insertBubbleText(mess,x,y,bubble,1.5)
            self.bubbleRegion.removeDrawable(clip)
            self.regions.append(editor)
            self.studioRegion.addDrawable(clip,self.savedRegion)
            i += 1
            
    def countNeutrals(self):
        return self.emotions.count('none')

    def any(self,boolList):
        for isTrue in boolList:
            if(isTrue):
                return True
        return False
   
    def all(self,boolList):
        for isTrue in boolList:
            if(not isTrue):
                return False
        return True

    def processFrame(self,cvImg,face=True):
        if(face):
            faces = faceCascade.detectMultiScale(cvImg)
            largestFace, area = None,0
            for x,y,w,h in faces:
                if(largestFace == None or w*h > area):
                    largestFace, area = (x,y,w,h), w*h
            if(largestFace != None and area > 10000):
                fX,fY,fW,fH = largestFace
                #cv2.rectangle(cvImg,(fX,fY),(fX+fW,fY+fH),(0,255,0),10) # face is thicker
                mouths = mouthCascade.detectMultiScale(cvImg[fY+3*fH//4:fY+fH,fX:fX+fW])
                largestMouth, area = None, 0
                for mx,my,mw,mh in mouths:
                    if(largestMouth == None or mw*mh > area):
                        largestMouth, area = (mx,my,mw,mh), mw*mh
                if(largestMouth != None):
                    mX,mY,mW,mH = largestMouth
                    #cv2.rectangle(cvImg,(fX+mX,fY+3*fH//4+mY),(fX+mX+mW,fY+3*fH//4+mY+mH),(0,255,0),5)
                    leftX, rightX, pointY = fX-50,fX+fW+50,fY+3*fH//4+mY-20
                    
                    currIm = cvToPIL(cvImg)
                    w, h = currIm.size[0], currIm.size[1]
                    newClip = Clip(currIm,cvImg,0,0,w,h,editor=[])
        
                    if(not self.placeSides(newClip,leftX,rightX,pointY)):
                        #print('failed to add to the sides')
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
                w,h = bubble.img.size[0]*self.bubbleRatio, bubble.img.size[1]*self.bubbleRatio
                #w,h = origW*self.bubbleRatio, origH*self.bubbleRatio #origW,origH # number of pixels in w,h on final
                x0,y0 = int(leftX-w),int(pointY-h)
                #print(f'w,h:{w,h}; leftX,pointY: {leftX,pointY}: x0,y0:{x0,y0}')
                
                bubble.x, bubble.y = x0*clip.editor.scale-50, y0*clip.editor.scale-50
                #print(f'w,h: {origW,origH}, and scaled: {w,h}')
                #print(f'bubble x/y on the left: {bubble.x,bubble.y}')
                
                clip.editor.addDrawable(bubble,clip.editor)
                
                scaledGraphic = bubble.img.resize((int(bubble.img.size[0]*self.bubbleRatio),int(bubble.img.size[1]*self.bubbleRatio)))
                mask = pilToCV(scaledGraphic)
                #x1 = int((bubble.x-clip.editor.x0)/clip.editor.scale*1 - clip.editor.oMarg)
                #y1 = int((bubble.y-clip.editor.y0)/clip.editor.scale - clip.editor.headerMarg)
                
                #print(f'top left: {x0},{y0}, bot right: {x0+w,y0+h}, limit: {clip.origImg.shape[1],clip.origImg.shape[0]}')
                #if(0 <= x1 and x1+mask.shape[1] <= clip.origImg.shape[1] and 0 <= y1 and y1+mask.shape[0] <= clip.origImg.shape[0]):
                x0,y0 = int((bubble.x-clip.editor.x0)/clip.editor.scale*1 - clip.editor.oMarg),int((bubble.y-clip.editor.y0)/clip.editor.scale - clip.editor.headerMarg)
                if(0 <= x0 and x0+w <= clip.origImg.shape[1] and 0 <= y0 and y0+h <= clip.origImg.shape[0]):
                    if(clip.editor.updateGraphics()):
                        #print('shouldve added to left succesfully!')
                        return True
                else:
                    if(bubble in clip.editor.drawables[1:]):
                        clip.editor.removeDrawable(bubble)
                    #print(f'failed to add {bubble.name}to the left')
            elif(bubble.direct == 'right'):
                w,h = bubble.img.size[0]*self.bubbleRatio, bubble.img.size[1]*self.bubbleRatio
                #w,h = self.bubbleRatio*origW/clip.scale, self.bubbleRatio*origH/clip.scale #origW,origH #
                #print(f'w,h:{w,h}; leftX,pointY: {leftX,pointY}: x0,y0:{x0,y0}')
                x0,y0 = int(rightX),int(pointY-h)
                
                bubble.x, bubble.y = x0*clip.editor.scale+50, y0*clip.editor.scale-50
                #print(f'bubble x/y on the right: {bubble.x,bubble.y}')
                
                clip.editor.addDrawable(bubble,clip.editor)
                
                scaledGraphic = bubble.img.resize((int(bubble.img.size[0]*self.bubbleRatio),int(bubble.img.size[1]*self.bubbleRatio)))
                mask = pilToCV(scaledGraphic)
                x0,y0 = int((bubble.x-clip.editor.x0)/clip.editor.scale*1 - clip.editor.oMarg),int((bubble.y-clip.editor.y0)/clip.editor.scale - clip.editor.headerMarg)
                if(0 <= x0 and x0+w <= clip.origImg.shape[1] and 0 <= y0 and y0+h <= clip.origImg.shape[0]):
                    if(clip.editor.updateGraphics()):
                        #print('shouldve added to right succesfully!')
                        return True
                else:
                    if(bubble in clip.editor.drawables[1:]):
                        clip.editor.removeDrawable(bubble)
                    #print(f'failed to add {bubble.name}to the right')
        #print('failed to add to both sides')
        return False

    def placePlain(self,clip):
        bubbs = self.bubbles.copy()
        random.shuffle(bubbs)
        for bubble in bubbs: # bubble is a clip object graphic
            if(bubble.direct == 'norm'):
                #origW,origH = bubble.w, bubble.h
                #w,h = origW/clip.scale, origH/clip.scale
                
                bubble.x, bubble.y = 50, 50
                #print('trying the norm:')
                
                clip.editor.addDrawable(bubble,clip.editor)
                scaledGraphic = bubble.img.resize((int(bubble.img.size[0]*self.bubbleRatio),int(bubble.img.size[1]*self.bubbleRatio)))
                mask = pilToCV(scaledGraphic)
                x1 = int((bubble.x-clip.editor.x0)/clip.editor.scale*1 - clip.editor.oMarg)
                y1 = int((bubble.y-clip.editor.y0)/clip.editor.scale - clip.editor.headerMarg)
                
                #print(mask.shape)
                if(5 <= x1 and x1+mask.shape[1] <= clip.origImg.shape[1]-5 and 5 <= y1 and y1+mask.shape[0] <= clip.origImg.shape[0]-5):
                    clip.editor.updateGraphics()
                    return True
                else:
                    if(bubble in clip.editor.drawables[1:]):
                        clip.editor.removeDrawable(bubble)
                    #print(f'failed to add {bubble.name}to the normal')

    ########################################################################
    ################## View  ###############################################
    ########################################################################
    
    def drawBoard(self,canvas):
        if(self.editing):
            self.savedRegion.draw(canvas)
            self.currEditorRegion.draw(canvas,False)
            for region in self.graphicsRegions:
                if(region.active):
                    region.draw(canvas,False)
        else:
            for region in self.regions:
                if(region.active):
                    region.draw(canvas)
            
    def drawButtons(self,canvas):
        for butt in self.filterButtons:
            butt.draw(canvas)
        for butt in self.controls:
            if(self.editing or butt.label != 'undo'):
                butt.draw(canvas)
        for butt in self.graphicButtons:
            butt.draw(canvas)

    def drawFrame(self,canvas):
        if(self.enterText):
            canvas.create_image((self.pBarLeft+self.width)//2, (self.gBarTop + self.height+80)//2,image=ImageTk.PhotoImage(self.blueBorder))
        canvas.create_image(self.width//2, self.height//2,image=ImageTk.PhotoImage(self.backgroundImage))

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
            canvas.create_text(x,y,fill='gray',text=f'TEXT SIZE: {int(self.textSize*12)}\nPress up/down arrows to change.\nPress enter to save your text.',font=self.small,anchor='nw')
        else:
            canvas.create_text(x,y,fill='gray',text=f'when you write in text bubbles, \nyour text will appear here!',font=self.small,anchor='nw')
            
    def redrawAll(self, canvas):
        self.drawFrame(canvas)
        self.drawButtons(canvas)
        self.drawBoard(canvas)
        self.drawLabels(canvas)
        self.drawTextBox(canvas)
        if(self.congrats):
            canvas.create_text(200,self.height//15,fill='gray',text=f'{self.title} was saved to myComicStrip.png, go take a look!',font=self.large,anchor='nw')


# Use of modes, MyModalApp structure taken from
# https://www.cs.cmu.edu/~112/notes/notes-animations-part3.html#subclassingModalApp
class MyModalApp(ModalApp):
    def appStarted(app):
        removeTempFiles('graphics')
        app.splashScreenMode = SplashScreenMode()
        app.gameMode = Studio()
        app.helpMode = HelpMode()
        app.aboutMode = AboutMode()
        app.editorHelpMode = AltHelpMode()
        app.setActiveMode(app.splashScreenMode)
        app.timerDelay = 70
        app.mouseMovedDelay = 70

def playGame():
    MyModalApp(width = 1200, height = 800)

def main():
    playGame()

if __name__ == '__main__':
    main()