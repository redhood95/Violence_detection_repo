import os 
import random 
import cv2 
import numpy as np 
from dataset.hockey import Hockey_data


Hockey = Hockey_data()

positvies, negatives = Hockey.load_data("data\HockeyFights\extracted_frames")

All_vids = np.concatenate(positvies,negatives)

#generating annotation

anno = []

for i in range(0,len(All_vids)):
    if i < (len(All_vids)/2):
        append(1)
    else :
        append(0)

anno = np.array(anno)


