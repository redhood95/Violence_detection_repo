import sys
import warnings
import time
import os.path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(1, 'D:\\res2020\Computer_Vision\\violence_detect\\root')
from sklearn.preprocessing import OneHotEncoder

from dataset.hockey import Hockey_data

class Load:
    def load(self):
        if os.path.isfile('arrays_dump/train_x.npy'):
            print ("File exist")
            train_x= np.load("arrays_dump/train_x.npy")
            train_y=np.load("arrays_dump/train_y.npy")
            eval_x=np.load("arrays_dump/eval_x.npy")
            eval_y=np.load("arrays_dump/eval_y.npy")
            test_x=np.load("arrays_dump/test_x.npy")
            test_y=np.load("arrays_dump/test_y.npy")

        else:
            print ("File not exist")
            Hockey = Hockey_data()

            train_x , train_y,eval_x , eval_y, test_x  , test_y = Hockey.load_data("data\HockeyFights\extracted_frames")

            np.save("arrays_dump/train_x.npy",train_x)
            np.save("arrays_dump/train_y.npy",train_y)
            np.save("arrays_dump/eval_x.npy",eval_x)
            np.save("arrays_dump/eval_y.npy",eval_y)
            np.save("arrays_dump/test_x.npy",test_x)
            np.save("arrays_dump/test_y.npy",test_y)

        onehot_encoder = OneHotEncoder(sparse=False)
        train_y = onehot_encoder.fit_transform(train_y)
        test_y = onehot_encoder.fit_transform(test_y)
        eval_y = onehot_encoder.fit_transform(eval_y)
        return train_x , train_y,eval_x , eval_y, test_x  , test_y