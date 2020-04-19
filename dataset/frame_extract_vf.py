import os 
import cv2

path  = "..\data\violentflows\movies"


os.mkdir("..\data\violentflows\extracted_frames")
os.mkdir("..\data\violentflows\extracted_frames\yes")
os.mkdir("..\data\violentflows\extracted_frames\ono")

for i in range(0,5):
    no_path = os.path.join(os.path.join(path,str(i)),"NonViolence")
    yes_path = os.path.join(os.path.join(path,str(i)),"Violence")

    yes =  os.listdir(yes_path)
    no  =  os.listdir(no_path)


