""" read in figure, scan it, get lots of small images to be analysed whether they contain a character/number"""

import numpy as np
import cv2
import sys

# idendify rekkari.jpeg gives 259X194


# Load an color image in grayscale
#img = cv2.imread('rekkari.jpeg',0)

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# load and show video
#cap = cv2.VideoCapture('/home/mka/Downloads/test.mp4')
cap = cv2.VideoCapture(sys.argv[1])
frames = []

    

while(cap.isOpened()):
    #try:
    ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame',gray)
    frame = cv2.transpose(frame)
    frame = cv2.flip(frame, flipCode=1)
    ok_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame',ok_color)        
    #except:
    #    break

    try:
        #frames.append(gray)
        frames.append(ok_color)
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

print("ok")
i=0
min=0
max=len(frames)
while (2>1):
    if  cv2.waitKey(1) & 0xFF == ord('p'):
        if i < max:
            i=i+1
            print("ip", i)
    elif cv2.waitKey(1) & 0xFF == ord('m'):
        if i > 0:
            i = i - 1
            print("im",i)
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('frame'+str(i),frames[i])
cv2.destroyAllWindows()
