"""
# python3 ./flip.py -infiles '*jpg'

if y dimension is bigger than x, flip dimensions

"""


import argparse
import glob
import os
import shutil
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-infiles', action='store', dest='infiles',
                    help='filenames of negative images')
#parser.add_argument('-width', action='store', dest='width',
#                    help='width of image in pixels', default=80, type=int)
#parser.add_argument('-height', action='store', dest='height',
#                    help='height of image in pixels', default=20, type=int)
args = parser.parse_args()


names=glob.glob(args.infiles)
print(names)

#generate slices of original pictures (to be negative samples)
for i, name in enumerate(names):
    img = cv2.imread(name,0)
    if (img.shape[0] > img.shape[1]):
        # flip and write
        rows,cols = img.shape
        rotated = np.rot90(img)
        cv2.imwrite(name, rotated)

