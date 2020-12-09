****************************
README
****************************
Welcome to the Comic Sans Artist! This is an application with a studio-style
user interface where you can build your own comic strip by taking pictures of 
yourself. Customize your comic and art style using different comic graphics, 
text bubbles and image filters. If you're feeling lazy, there's an option to 
auto-generate a comic from taking a video of yourself; we'll make a (bad) 
guess of what you're trying to say with some text bubbles using facial and
emotion recognition.

Run this program from your terminal by running main.py using the command
'python3 main.py'. Libraries which need to be installed:
- opencv-python, numpy, pillow(PIL), requests, keras, tensorflow
- pandas, adam, cinit, kwargs(these four shouldn't need to be downloaded to run, since they were only used to train the emotion recognition model)

From there, the application should
be fairly straightforward to use; there is a help menu for reference as well.

PLEASE NOTE: the program takes a second to load before the application pops up; this is
due to the pre-loading of the emotion recognition model and some facial
recognition cascades.

Other notes, if for whatever reason things are not intuitive enough(sorry!):
- the filters will only apply if there are clips in the studio or editor;
it cannot filter nothing unfortunately
- click "snap photo" to take video/photos