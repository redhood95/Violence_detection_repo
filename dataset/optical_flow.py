import os 
import cv2


# for hockey fights # 

Hockey_path  = "..\data\HockeyFights\extracted_frames"

H_list  = os.listdir(Hockey_path)
for category in H_list:
    frame_folders = os.listdir(os.path.join(Hockey_path,category))
    for seq in frame_folders:
        frame_path = os.path.join(os.path.join(Hockey_path,category),seq)
        frames = os.listdir(frame_path)
        for frame_num in range(1,len(frames)-1):
            #load frame 1#
            frame1 = cv2.imread(frame_path+"\\"+str(frame_num)+".png")
            frame2 = cv2.imread(frame_path+"\\"+str(frame_num+1)+".png")
            hsv = np.zeros_like(frame1)
            hsv[...,1] = 255
            flow = cv.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)


# for violent flows # 
Vf_path  = "..\data\\violentflows\extracted_frames"
vf_list  = os.listdir(Vf_path)
for category in vf_list:
    frame_folders = os.listdir(os.path.join(Vf_path,category))
    for seq in frame_folders:
        frame_path = os.path.join(os.path.join(Vf_path,category),seq)
        frames = os.listdir(frame_path)
        for frame_num in range(1,len(frames)-1):
            #load frame 1#
            frame1 = cv2.imread(frame_path+"\\"+str(frame_num)+".png")
            frame2 = cv2.imread(frame_path+"\\"+str(frame_num+1)+".png")
            hsv = np.zeros_like(frame1)
            hsv[...,1] = 255
            flow = cv.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)


