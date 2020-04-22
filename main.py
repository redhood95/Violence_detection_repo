import os 
import random 
import cv2 
import numpy as np 
from dataset.loader import Load
from framework.train_keras import Train_using_keras
from sklearn.metrics import confusion_matrix



Loader = Load()
train_x , train_y,eval_x , eval_y, test_x  , test_y = Loader.load()



trainer = Train_using_keras()
## Use for Training model
# trainer.train(X=train_x,Y=train_y,X_eval=eval_x,Y_eval=eval_y,seq_length = 48, model='lrcn', batch_size=32, nb_epoch=10)


##Use for Testing model


prediction = trainer.test(Path_to_model='saved_models/lrcn_10_v1.json',Path_to_weights="saved_models/lrcn_10_v1.h5",X_test=test_x)

results = []
for i in prediction:
    if  i[0] > 0.50:
        results.append(int(0))
    else:
        results.append(int(1))

cm = confusion_matrix(test_y, results)

print(cm)

