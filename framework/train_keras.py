import sys
import warnings
import time
import os.path
warnings.filterwarnings("ignore")
sys.path.insert(1, 'D:\\res2020\Computer_Vision\\violence_detect\\root')
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models.keras_models import ResearchModels
class train_using_keras:

