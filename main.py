import os 
import random 
import cv2 
import numpy as np 
from dataset.hockey import Hockey_data


Hockey = Hockey_data()

Hockey.load_data("data\HockeyFights\extracted_frames")
