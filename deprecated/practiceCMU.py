from cmu_112_graphics import *

def appStarted(app):
    app.message = 'Press any key'

def keyPressed(app, event):
    app.message = f"event.key == '{event.key}'"

def redrawAll(app, canvas):
    canvas.create_text(app.width/2, 40, text=app.message, font='Arial 30 bold')
    
    keyNamesText = '''Here are the legal event.key names:
                      * Keyboard key labels (letters, digits, punctuation)
                      * Arrow directions ('Up', 'Down', 'Left', 'Right')
                      * Whitespace ('Space', 'Enter', 'Tab', 'Backspace')
                      * Other commands ('Delete', 'Escape')'''

    y = 80
    for line in keyNamesText.splitlines():
        canvas.create_text(app.width/2, y, text=line.strip(), font='Arial 20')
        y += 30

runApp(width=600, height=400)