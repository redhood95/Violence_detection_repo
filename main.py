import os 
import random 
import cv2 
import numpy as np 
from dataset.loader import Load
from framework.train_keras import Train_using_keras

Loader = Load()
train_x , train_y,eval_x , eval_y, test_x  , test_y = Loader.load()


trainer = Train_using_keras()
trainer.train(X=train_x,Y=train_y,X_eval=eval_x,Y_eval=eval_y,seq_length = 48, model='lrcn', batch_size=32, nb_epoch=10)