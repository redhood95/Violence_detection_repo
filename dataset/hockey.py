import os 
import random 
import cv2 
import numpy as np 
class Hockey:
    y = "..\data\HockeyFights\yes"
    n = "..\data\HockeyFights\ono"
    yes =  os.listdir(y)
    no  =  os.listdir(n)
    paths = []
    for i in yes:
        paths.append([os.path.join(y,i),'yes'])

    for i in no:
        paths.append([os.path.join(n,i),'no'])


    res = random.sample(paths, len(paths))

    def load_data(self):
        print("total videos " + str(len(self.res)))
        x = []
        y = []
        vid_count = 0
        all = []
        for index in self.res:
            vid_count = vid_count +1
            print("loading video............... no:"+str(vid_count))
            video = [] 
            vid_path = index[0]
            annotation = index[1]
            y.append(annotation)
            cap = cv2.VideoCapture(vid_path)
            count = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                
                if ret == True:
                    count = count + 1
                    if count > 41 :
                        break
                    video.append(frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            print("number of frames loaded: "+str(len(video)))
            np_vid = np.array(video)
            all.append(np_vid)
            cap.release()
        
        print("number of videos loaded: "+str(len(all)))
        x = np.array(all)
        y =  np.array(y)
        return x, y


        

h = Hockey()
x,y = h.load_data()

print(x.shape)
print(y.shape)

y.reshape(1000,1)
print(y.shape)

x.reshape(1000,41,288,360,3)
print(x.shape)