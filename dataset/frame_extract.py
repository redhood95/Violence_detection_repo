import os 
import cv2

dataset = "HockeyFights" # or "violentflows"

y = "..\data\HockeyFights\yes"
n = "..\data\HockeyFights\ono"

yes =  os.listdir(y)
no  =  os.listdir(n)

os.mkdir("..\data\HockeyFights\extracted_frames")
os.mkdir("..\data\HockeyFights\extracted_frames\yes")
os.mkdir("..\data\HockeyFights\extracted_frames\ono")


for i in range(0,len(yes)):
    print("yes===================" +str(i) )
    folder_path = os.path.join("..\data\HockeyFights\extracted_frames\yes",str(i))
    os.mkdir(folder_path)
    vid_path = os.path.join(y,yes[i])
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            count = count + 1
            image_path = os.path.join(folder_path,str(count))
            cv2.imwrite(image_path+".png",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    print("number of frames extracted :"+str(count))
    cap.release()



for i in range(0,len(no)):
    folder_path = os.path.join("..\data\HockeyFights\extracted_frames\ono",str(i))
    os.mkdir(folder_path)
    vid_path = os.path.join(n,no[i])
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            count = count + 1
            image_path = os.path.join(folder_path,str(count))
            cv2.imwrite(image_path+".png",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    print("number of frames extracted :"+str(count))
    cap.release()