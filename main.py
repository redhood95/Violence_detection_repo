import os 
import random 
import cv2 
import numpy as np 
from dataset.hockey import Hockey_data
from sklearn.model_selection import train_test_split


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


train_x , x , train_y , y = train_test_split(All_vids , anno , 
                                            test_size = 0.2 ,
                                            random_state = 111)

eval_x , test_x , eval_y , test_y = train_test_split(x , y , 
                                                    test_size = 0.5 , 
                                                    random_state = 111)
