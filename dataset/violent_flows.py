import os 
import cv2 
import numpy as np 
from sklearn.model_selection import train_test_split


class Violent_flows:
    def load_data(self,path):
        v_list  = os.listdir(path)

        y_vid = []
        n_vid = []
        #loading original sequence
        for category in v_list:
            print(category)
            frame_folders = os.listdir(os.path.join(path,category))
            for seq in frame_folders:
                print(seq)
                frame_path = os.path.join(os.path.join(path,category),seq)
                frames = os.listdir(frame_path)
                num_frames_present = len(frames)
                for frame_num in range(1,len(frames)-1):
                    #load frame 1#
                    frame1 = cv2.imread(frame_path+"\\"+str(frame_num)+".png")
                    frame1 = cv2.resize(frame1,(50,100))
                    if category == v_list[0]:
                        y_vid.append(frame1)
                    elif category == v_list[1]:
                        n_vid.append(frame1)

        y_vid = np.array(y_vid)
        n_vid = np.array(n_vid)
        print(len(y_vid))
        print(len(n_vid))
        
        print('original sequence loaded')

   


        # train_x , x , train_y , y = train_test_split(All_vids , anno , 
        #                                     test_size = 0.2 ,
        #                                     random_state = 324)

        # eval_x , test_x , eval_y , test_y = train_test_split(x , y , 
        #                                                     test_size = 0.5 , 
        #                                                     random_state = 324)

        # return train_x,train_y,eval_x  , eval_y, test_x , test_y


vf = Violent_flows()

vf.load_data('../data/violentflows/extracted_frames')