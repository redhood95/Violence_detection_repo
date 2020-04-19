import os 
import cv2

path  = "..\data\violentflows\movies"


os.mkdir("..\data\violentflows\extracted_frames")
os.mkdir("..\data\violentflows\extracted_frames\yes")
os.mkdir("..\data\violentflows\extracted_frames\ono")

for i in range(0,5):
    print("==================================" + str(i)+"==================================")
    no_path = os.path.join(os.path.join(path,str(i)),"NonViolence")
    yes_path = os.path.join(os.path.join(path,str(i)),"Violence")

    yes =  os.listdir(yes_path)
    no  =  os.listdir(no_path)

    for k in range(0,len(yes)):
        print("yes===================" +str(k) )
        folder_path = os.path.join("..\data\violentflows\extracted_frames\yes",str(i)+"__"+str(k))
        os.mkdir(folder_path)
        vid_path  = os.path.join(yes_path,yes[k])
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




    for k in range(0,len(no)):
        print("no===================" +str(k) )



    


