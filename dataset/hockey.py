import os 
import random 
import cv2 
import numpy as np 
from sklearn.model_selection import train_test_split

class Hockey_data:

    def load_data(self,Hockey_path):


        # Hockey_path  = "..\data\HockeyFights\extracted_frames"

        H_list  = os.listdir(Hockey_path)
        # Y_Vids = []
        # N_Vids = []
        All_vids = []
        #loading original sequence
        for category in H_list:

            frame_folders = os.listdir(os.path.join(Hockey_path,category))
            for seq in frame_folders:
                frame_path = os.path.join(os.path.join(Hockey_path,category),seq)
                frames = os.listdir(frame_path)
                num_frames_present = len(frames)
                vid = []
                for frame_num in range(1,len(frames)-1):
                    #load frame 1#
                    frame1 = cv2.imread(frame_path+"\\"+str(frame_num)+".png")
                    frame1 = cv2.resize(frame1,(50,100))
                    vid.append(frame1)
                empty = np.zeros_like(frame1)
                for i in range(num_frames_present,50):
                    vid.append(empty)

                All_vids.append(vid)
                # if category == 'ono':                    
                #     N_Vids.append(vid)                    
                # elif category == 'yes':
                #     Y_Vids.append(vid)

        print('original sequence loaded')
        #generating annotation

        ## doing some data Augmentatation

        # #loading  sequence with gaussian blur
        # for category in H_list:
        #     frame_folders = os.listdir(os.path.join(Hockey_path,category))
        #     for seq in frame_folders:
        #         frame_path = os.path.join(os.path.join(Hockey_path,category),seq)
        #         frames = os.listdir(frame_path)
        #         num_frames_present = len(frames)
        #         vid = []
        #         for frame_num in range(1,len(frames)-1):
        #             #load frame 1#
        #             frame1 = cv2.imread(frame_path+"\\"+str(frame_num)+".png")
        #             frame1 = cv2.GaussianBlur(cv2.resize(frame1,(250,350)),(5,5),0)
        #             vid.append(frame1)
        #         empty = np.zeros_like(frame1)
        #         for i in range(num_frames_present,50):
        #             vid.append(empty)
        #         if category == 'ono':
        #             N_Vids.append(vid)
        #         elif category == 'yes':
        #             Y_Vids.append(vid)

        # print('Gaussian sequence loaded')

        # y_array = np.array(Y_Vids)
        # n_array = np.array(N_Vids)
        All_vids = np.array(All_vids)
        anno = []

        for i in range(0,len(All_vids)):
            if i < (len(All_vids)/2):
                anno.append(0)
            else :
                anno.append(1)

        anno = np.array(anno)


        train_x , x , train_y , y = train_test_split(All_vids , anno , 
                                            test_size = 0.2 ,
                                            random_state = 324)

        eval_x , test_x , eval_y , test_y = train_test_split(x , y , 
                                                            test_size = 0.5 , 
                                                            random_state = 324)

        return train_x,train_y,eval_x  , eval_y, test_x , test_y

        
