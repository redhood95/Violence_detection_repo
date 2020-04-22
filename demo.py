import warnings
import time
import os.path
warnings.filterwarnings("ignore")
import cv2 
from keras.models import model_from_json
import numpy as np




path_to_video = 'data/HockeyFights/yes/fi493_xvid.avi'


cap = cv2.VideoCapture(path_to_video)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('14.mp4',fourcc, 10, (frame_width,frame_height))

vid_org=[]
vid = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        vid_org.append(frame)
        screen = cv2.resize(frame, (50,100))
        vid.append(screen)
        cv2.imshow('window',cv2.resize(frame,(50,100)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    else:
        break

cap.release()
# out.release()
cv2.destroyAllWindows()

empty = np.zeros_like(vid[0])
for i in range(len(vid),48):
    vid.append(empty)


original = np.array(vid_org)
test_vid = np.array(vid)
test_vid = np.reshape(test_vid,(1,48,100,50,3))

print(original.shape)
print(test_vid.shape)

##Loading model##
json_file = open('saved_models/lrcn_10_v1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/lrcn_10_v1.h5")
print("Loaded model from disk")

pred = loaded_model.predict(test_vid)
print(pred)

if pred[0][0]<0.50:
    res = 'Yes'
else:
    res = 'NO'
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

for i in original:
    i = cv2.putText(i, res, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)

    out.write(i)

out.release()
