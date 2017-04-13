"""
# python3 subSampleRandomNegatives.py -infiles '*jpg' -nrandom 20000

~/PycharmProjects/Rekkari/Training/Negative/NegativesSmall
python3 ../../../subSampleRandomNegatives.py -infiles '*jpg' -nrandom 20000


Check with 
opencv_createsamples -w 80 -h 20 -vec merged.vec
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
parser.add_argument('-nrandom', action='store', dest='nrandom',
                    help='number of random files to be copied', default=10000, type=int)
args = parser.parse_args()

cwd = os.getcwd()
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


