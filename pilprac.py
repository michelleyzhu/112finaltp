# SideScroller3:

# Now with walls that track when you run into them (but
# ignore while you are still crossing them).

from cmu_112_graphics import *

class SideScroller3(App):
    def appStarted(self):
        self.scrollX = 0 
        self.scrollMargin = 50
        self.playerX = self.scrollMargin
        self.playerY = 0
        self.playerWidth = 10
        self.playerHeight = 20
        self.walls = 5
        self.wallPoints = [0]*self.walls
        self.wallWidth = 20
        self.wallHeight = 40
        self.wallSpacing = 90 # wall left edges are at 90, 180, 270,...
        self.currentWallHit = -1 # start out not hitting a wall

    def getPlayerBounds(self):
        # returns absolute bounds, not taking scrollX into account
        (x0, y1) = (self.playerX, self.height/2 - self.playerY)
        (x1, y0) = (x0 + self.playerWidth, y1 - self.playerHeight)
        return (x0, y0, x1, y1)

    def getWallBounds(self, wall):
        # returns absolute bounds, not taking scrollX into account
        (x0, y1) = ((1+wall) * self.wallSpacing, self.height/2)
        (x1, y0) = (x0 + self.wallWidth, y1 - self.wallHeight)
        return (x0, y0, x1, y1)

    def getWallHit(self):
        # return wall that player is currently hitting
        # note: this should be optimized to only check the walls that are visible
        # or even just directly compute the wall without a loop
        playerBounds = self.getPlayerBounds()
        for wall in range(self.walls):
            wallBounds = self.getWallBounds(wall)
            if (self.boundsIntersect(playerBounds, wallBounds) == True):
                return wall
        return -1

    def boundsIntersect(self, boundsA, boundsB):
        # return l2<=r1 and t2<=b1 and l1<=r2 and t1<=b2
        (ax0, ay0, ax1, ay1) = boundsA
        (bx0, by0, bx1, by1) = boundsB
        return ((ax1 >= bx0) and (bx1 >= ax0) and
                (ay1 >= by0) and (by1 >= ay0))

    def checkForNewWallHit(self):
        # check if we are hitting a new wall for the first time
        wall = self.getWallHit()
        if (wall != self.currentWallHit):
            self.currentWallHit = wall
            if (wall >= 0):
                self.wallPoints[wall] += 1

    def makePlayerVisible(self):
        # scroll to make player visible as needed
        if (self.playerX < self.scrollX + self.scrollMargin):
            self.scrollX = self.playerX - self.scrollMargin
        if (self.playerX > self.scrollX + self.width - self.scrollMargin):
            self.scrollX = self.playerX - self.width + self.scrollMargin

    def movePlayer(self, dx, dy):
        self.playerX += dx
        self.playerY += dy
        self.makePlayerVisible()
        self.checkForNewWallHit()

    def sizeChanged(self):
        self.makePlayerVisible()

    def mousePressed(self, event):
        self.playerX = event.x + self.scrollX
        self.checkForNewWallHit()

    def keyPressed(self, event):
        if (event.key == "Left"):    self.movePlayer(-5, 0)
        elif (event.key == "Right"): self.movePlayer(+5, 0)
        elif (event.key == "Up"):    self.movePlayer(0, +5)
        elif (event.key == "Down"):  self.movePlayer(0, -5)

    def redrawAll(self, canvas):
        # draw the base line
        lineY = self.height/2
        lineHeight = 5
        canvas.create_rectangle(0, lineY, self.width, lineY+lineHeight,fill="black")

        # draw the walls
        # (Note: should optimize to only consider walls that can be visible now!)
        sx = self.scrollX
        for wall in range(self.walls):
            (x0, y0, x1, y1) = self.getWallBounds(wall)
            fill = "orange" if (wall == self.currentWallHit) else "pink"
            canvas.create_rectangle(x0-sx, y0, x1-sx, y1, fill=fill)
            (cx, cy) = ((x0+x1)/2 - sx, (y0 + y1)/2)
            canvas.create_text(cx, cy, text=str(self.wallPoints[wall]))
            cy = lineY + 5
            canvas.create_text(cx, cy, text=str(wall), anchor=N)

        # draw the player
        (x0, y0, x1, y1) = self.getPlayerBounds()
        canvas.create_oval(x0 - sx, y0, x1 - sx, y1, fill="cyan")

        # draw the instructions
        msg = "Use arrows to move, hit walls to score"
        canvas.create_text(self.width/2, 20, text=msg)

SideScroller3(width=300, height=300)