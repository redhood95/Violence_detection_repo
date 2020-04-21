import os 
import random 
import cv2 
import numpy as np 
from dataset.hockey import Hockey_data


if os.path.isfile('array_dump/train_x.npy'):
    print ("File exist")
    train_x= np.load("array_dump/train_x.npy")
    train_y=np.load("array_dump/train_y.npy")
    eval_x=np.load("array_dump/eval_x.npy")
    eval_y=np.load("array_dump/eval_y.npy")
    test_x=np.load("array_dump/test_x.npy")
    test_y=np.load("array_dump/test_y.npy")

else:
    print ("File not exist")
    Hockey = Hockey_data()

    train_x , train_y,eval_x , eval_y, test_x  , test_y = Hockey.load_data("data\HockeyFights\extracted_frames")

    np.save("array_dump/train_x.npy",train_x)
    np.save("array_dump/train_y.npy",train_y)
    np.save("array_dump/eval_x.npy",eval_x)
    np.save("array_dump/eval_y.npy",eval_y)
    np.save("array_dump/test_x.npy",test_x)
    np.save("array_dump/test_y.npy",test_y)




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