"""
# python3 negatives2Negatives.py -infiles '*jpg' -outdir './Negatives' -width 32 -height 48 -num 1000

pick random subfigures from figures and store them
subfigure has width of parameters width and height of parameter height 

"""


import argparse
import glob
import os
#import shutil
#import cv2
#import sys
import numpy as np
import random
from shutil import copy

parser = argparse.ArgumentParser()

parser.add_argument('-infiles', action='store', dest='infiles',
                    help='filenames of negative images')
parser.add_argument('-outdir', action='store', dest='outdir',
                    help='directory for new negative images')
parser.add_argument('-width', action='store', dest='width', 
                    help='width of image in pixels', default='32')
parser.add_argument('-height', action='store', dest='height', 
                    help='height of image in pixels', default='48')
parser.add_argument('-num', action='store', dest='num', type=integer,
                    help='number of distorted images to generate', default='1000')
args = parser.parse_args()
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

cwd = os.getcwd()
infiles = glob.glob(args.infiles)
i=0
while i < args.num:
    fname = random.choice(infiles)
    img = cv2.imread(fname, 0)
    y=random.randint(0, img.shape[0]-args.height-1)
    x=random.randint(0, img.shape[0]-args.height-1)
    out = img[y:y+args.height, x:x+args.width]
    cv2.imwrite(args.outdir+'/'+fname+str(i)+'.tif', out)
    

sampledir = cwd+'/Negativesrandom'+str(args.nrandom)

try:
    os.makedirs(sampledir)
except:
    pass

names=glob.glob(args.infiles)
names = random.sample(names, args.nrandom)
#print(names)

#copy a subset of samples
for name in names:
    copy(name, sampledir)


