# python3 ~/PycharmProjects/Rekkari/readMatPlotLib.py fin9.Plate.exp0.png fin9.Plate.exp0.box

import numpy as np
import cv2
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle
import sys

img = cv2.imread(sys.argv[1])
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

if len(sys.argv)> 2:
    lines = open(sys.argv[2]).readlines()
    rectangles = []
    for line in lines:
        rectangles.append([int(line.split()[1]), 
                          int(line.split()[2]),
                          int(line.split()[3]),
                           int(line.split()[4])])
    for rectangle in rectangles:
        print(rectangle[0], rectangle[1], rectangle[2], rectangle[3])
        plt.gca().add_patch(plt.Rectangle((rectangle[0], rectangle[1]),
                                          rectangle[2]-rectangle[0],
                                          rectangle[3]-rectangle[1],
                                          alpha=0.5,facecolor="#aaaaaa"))

    
plt.show()

