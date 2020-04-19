import os 
import random 
import cv2 
import numpy as np 

Hockey_path  = "..\data\HockeyFights\extracted_frames"

H_list  = os.listdir(Hockey_path)
for category in H_list:
    os.mkdir(os.path.join(save_folder,category))
    frame_folders = os.listdir(os.path.join(Hockey_path,category))
    for seq in frame_folders:
        os.mkdir(os.path.join(os.path.join(save_folder,category),seq))
        frame_path = os.path.join(os.path.join(Hockey_path,category),seq)
        frames = os.listdir(frame_path)
        for frame_num in range(1,len(frames)-1):
            #load frame 1#
            frame1 = cv2.imread(frame_path+"\\"+str(frame_num)+".png")