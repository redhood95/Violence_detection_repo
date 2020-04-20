import os 
import random 
import cv2 
import numpy as np 

Hockey_path  = "..\data\HockeyFights\extracted_frames"

H_list  = os.listdir(Hockey_path)
Y_Vids = []
N_Vids = []

#loading original sequence
for category in H_list:
    print(category)
    frame_folders = os.listdir(os.path.join(Hockey_path,category))
    for seq in frame_folders:
        print(seq)
        frame_path = os.path.join(os.path.join(Hockey_path,category),seq)
        frames = os.listdir(frame_path)
        num_frames_present = len(frames)
        vid = []
        for frame_num in range(1,len(frames)-1):
            #load frame 1#
            frame1 = cv2.imread(frame_path+"\\"+str(frame_num)+".png")
            frame1 = cv2.resize(frame1,(250,350))
            vid.append(frame1)
        empty = np.zeros_like(frame1)
        for i in range(num_frames_present,50):
            vid.append(empty)

        print(len(vid))
        if category == 'ono':
            print(category)
            N_Vids.append(vid)
            print('vid added')
        elif category == 'yes':
            print(category)
            Y_Vids.append(vid)
            print('vid added')

print('original sequence loaded')

## doing some data Augmentatation

#loading  sequence with gaussian blur
for category in H_list:
    print(category)
    frame_folders = os.listdir(os.path.join(Hockey_path,category))
    for seq in frame_folders:
        print(seq)
        frame_path = os.path.join(os.path.join(Hockey_path,category),seq)
        frames = os.listdir(frame_path)
        num_frames_present = len(frames)
        vid = []
        for frame_num in range(1,len(frames)-1):
            #load frame 1#
            frame1 = cv2.imread(frame_path+"\\"+str(frame_num)+".png")
            frame1 = cv.GaussianBlur(cv2.resize(frame1,(250,350)),(5,5),0)
            vid.append(frame1)
        empty = np.zeros_like(frame1)
        for i in range(num_frames_present,50):
            vid.append(empty)

        print(len(vid))
        if category == 'ono':
            print(category)
            N_Vids.append(vid)
            print('vid added')
        elif category == 'yes':
            print(category)
            Y_Vids.append(vid)
            print('vid added')

print('original sequence loaded')


y_array = np.array(Y_Vids)
n_array = np.array(N_Vids)

print(len(y_array))

print(y_array.shape)
print(len(n_array))

print(n_array.shape)

        
