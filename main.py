import os 
import random 
import cv2 
import numpy as np 
from dataset.hockey import Hockey_data



Hockey = Hockey_data()

train_x , train_y,eval_x , eval_y, test_x  , test_y = Hockey.load_data("data\HockeyFights\extracted_frames")


print(train_x.shape)
print(train_y.shape)
print(eval_x.shape)
print(eval_y.shape)
print(test_x.shape)
print(test_y.shape)
print(len(train_x))
print(len(train_y))
print(len(eval_x))
print(len(eval_y))
print(len(test_x))
print(len(test_y))