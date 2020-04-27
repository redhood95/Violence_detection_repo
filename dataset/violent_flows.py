import os 
import cv2 
import numpy as np 
from sklearn.model_selection import train_test_split


class Violent_flows:
    def load_data(self,path):
        v_list  = os.listdir(path)

        All_vids = []
        #loading original sequence
        for category in v_list:
            print(category)
            frame_folders = os.listdir(os.path.join(path,category))
            for seq in frame_folders:
                print(seq)
                frame_path = os.path.join(os.path.join(path,category),seq)
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

        print('original sequence loaded')

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


vf = Violent_flows()

vf.load_data('../data/violentflows/extracted_frames')