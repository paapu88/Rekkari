# python3 add_balls.py
# adds rectangles on top of given plates

"""
1)Read selected picture
2) make rectangle to it by mouse lef button
3) by right mouse button save selected rectangle in given accuracy in pixels,
4 go for the next picture (goto 1) ).

Finnish car number plate: 118 mm x 442 mm
try here 20x80 pixel
"""

import sys
from PyQt5.QtWidgets import (QMainWindow, QTextEdit, 
    QAction, QFileDialog, QApplication, QWidget, QVBoxLayout, QPushButton, QLabel)
from PyQt5.QtGui import QIcon
import numpy as np
import cv2
from matplotlib import pyplot as plt

class MouseRectangle():
    """ get a rectangle by mouse"""
    def __init__(self):
        #super().__init__()
        self.refPt = []
        self.cropping = False
        self.image = None

    def set_image(self, image):
        self.image = image

    def reset(self):
        self.refPt = []
        self.cropping = False

    def get_refPt(self):
        return self.refPt

    def plot_xy(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("xy", x, y)

    def click_and_crop(self,event, x, y, flags, param):
	    # grab references to the global variables

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            self.cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.refPt.append((x, y))
            self.cropping = False

            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (255, 255, 255), -2)
            cv2.imshow("image", self.image)


class Example(QWidget):


    def __init__(self):
        super().__init__()


        self.mouse = MouseRectangle()

        layout = QVBoxLayout()
        self.btn = QPushButton("QFileDialog static method demo")
        self.btn.clicked.connect(self.showDialog())

        layout.addWidget(self.btn)
        self.le = QLabel("Hello")

        layout.addWidget(self.le)
        self.btn1 = QPushButton("QFileDialog object")
        self.btn1.clicked.connect(self.showDialog())
        layout.addWidget(self.btn1)

        self.contents = QTextEdit()
        layout.addWidget(self.contents)
        self.setLayout(layout)
        self.setWindowTitle("File Dialog demo")
        
    def getNewName(self, oldname, subdir='img'):
        """
        get new filename with extra path
        """
        import os
        #mydir = os.path.dirname(oldname)
        mydir = os.getcwd()
        name = os.path.basename(oldname)
        #generate new dir if it doesnot exist
        newdir = mydir + '/'+ subdir
        print("newdir:", newdir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        return newdir+'/'+'sample_'+name

    def showDialog(self):
        import glob
        import math
        fnames =glob.glob('*jpg')
        fnames = fnames + glob.glob('*png')
        for fname in fnames:
            try:
                img = cv2.imread(fname,0)
                print("size:", img.shape[0], img.shape[1])
                print(fname)
                # cv2.namedWindow('image')

                if (img.shape[0] > 1000):
                    img = cv2.resize(img, (int(img.shape[0]/2), int(img.shape[1]/2)))
                clone = img.copy()
                self.mouse.set_image(image=img)
                cv2.imshow('image', img)

                #cv2.setMouseCallback('image', plot_xy)
                #cv2.setMouseCallback('image', self.mouse.plot_xy)
                cv2.setMouseCallback('image', self.mouse.click_and_crop)

                #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
                #plt.xticks([]), plt.yticks([])  # to hide tick values on X,Y axis
                #plt.show()

                # keep looping until the 'd or c' key is pressed
                mydelete = False
                myprint = False
                while True:
                    # display the image and wait for a keypress
                    cv2.imshow("image", img)
                    key = cv2.waitKey(1) & 0xFF

                    # if the 'r' key is pressed, reset the cropping region
                    if key == ord("r"):
                        img = clone.copy()
                        self.mouse.reset()
                        self.mouse.set_image(image=img)

                    # if the 'c' key is pressed, break from the loop
                    elif key == ord("c"):
                        break

                    # if the 'k' key is pressed, keep the original
                    elif key == ord("k"):
                        myprint = True
                        break                    

                    # if the 'd' key is pressed, do not write image (it is virtually deleted)
                    elif key == ord("d"):
                        mydelete = True
                        break                    

                # if there are two reference points, then crop the region of interest
                # from teh image and display it
                refPt = self.mouse.get_refPt()
                newname = self.getNewName(fname,'HumanProcessed')
                if not(mydelete):
                    print(newname)
                    if myprint:
                        print("writing original:", newname)
                        cv2.imwrite(newname,clone)
                    elif len(refPt) >= 2:
                        center = (int((refPt[0][0]+refPt[1][0])/2), int((refPt[0][1]+refPt[1][1])/2))
                        radius = int(0.5*math.sqrt(\
                                                   (refPt[1][1]-refPt[0][1])*\
                                                   (refPt[1][1]-refPt[0][1])+\
                                                   (refPt[1][0]-refPt[0][0])*\
                                                   (refPt[1][0]-refPt[0][0]))) 
                        print("writing with added rectangle :", newname)
                        #cv2.rectangle(clone, refPt[0], refPt[1], (255, 255, 255), -2)
                        cv2.circle(clone, center, radius, (255, 255, 255), thickness=-2)
                        cv2.imwrite(newname,clone)
                else:
                    print("not writing:", newname)
            except:
                print ("skipping: ", fname)

        # close all open windows
        cv2.destroyAllWindows()
               
        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
