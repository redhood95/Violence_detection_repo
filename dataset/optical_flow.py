import os 
import cv2
import numpy as np

# for hockey fights # 

Hockey_path  = "..\data\HockeyFights\extracted_frames"
save_folder = "..\data\HockeyFights\extracted_frames\optical_flow"
os.mkdir(save_folder)

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
            frame2 = cv2.imread(frame_path+"\\"+str(frame_num+1)+".png")
            hsv = np.zeros_like(frame1)
            frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            
            hsv[...,1] = 0
            flow = cv2.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('frame2',bgr)
            dummy = os.path.join(os.path.join(save_folder,category),seq)+'\\'+str(frame_num)+'.png'
            print(dummy)
            cv2.imwrite(dummy,bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break


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
            frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            
            hsv[...,1] = 0
            flow = cv2.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('frame2',bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break


