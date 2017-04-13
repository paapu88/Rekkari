""" read in figure, scan it, get lots of small images to be analysed whether they contain a character/number"""

import numpy as np
import cv2

# idendify rekkari.jpeg gives 259X194


# Load an color image in grayscale
#img = cv2.imread('rekkari.jpeg',0)

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# load and show video
cap = cv2.VideoCapture('/home/mka/Downloads/test.mp4')
frames = []
while(cap.isOpened()):
    ret, frame = cap.read()
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    except:
        break
cap.release()

myframe=frames[100]
cv2.line(myframe,(0,0),(511,511),(255,0,0),5)
cv2.rectangle(myframe,(384,0),(510,128),(0,255,0),3)
cv2.circle(myframe,(447,63), 63, (0,0,255), -1)

cv2.imshow('myframe',myframe)
cv2.waitKey(0)
cv2.destroyAllWindows()
