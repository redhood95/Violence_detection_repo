import os 
import random 
import cv2 
import numpy as np 

Hockey_path  = "..\data\HockeyFights\extracted_frames"

H_list  = os.listdir(Hockey_path)
Y_vids = []
N_Vids = []
for category in H_list:
    print(category)
    frame_folders = os.listdir(os.path.join(Hockey_path,category))
    for seq in frame_folders:
        print(seq)
        frame_path = os.path.join(os.path.join(Hockey_path,category),seq)
        frames = os.listdir(frame_path)
        vid = []
        for frame_num in range(1,len(frames)-1):
            #load frame 1#
            frame1 = cv2.imread(frame_path+"\\"+str(frame_num)+".png")
            vid.append(frame1)

    if category == 'ono':
        N_Vids.append(vid)
    elif category == 'yes':
        Y_Vids.append(vid)

y_array = np.array(Y_vids)
n_array = np.array(N_vids)

print(len(y_array))

print(y_array.shape)
print(len(n_array))

print(n_array.shape)

        
