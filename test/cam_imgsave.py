import numpy as np
from gym_ras.tool.keyboard import Keyboard
import cv2 as cv
from pathlib import Path

save_dir = "./data/cam_cal"
cap = cv.VideoCapture('/dev/video0')
if not cap.isOpened():
 print("Cannot open camera")
 exit()

savedir = Path(save_dir)
savedir.mkdir(parents=True, exist_ok=True)
cnt = 0
while True:
 # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    
    cv.imshow('frame', frame)
    ch = cv.waitKey(1)
    if ch == ord("q"):
       break
    elif ch == ord("s"):
       cnt+=1
       file = str(savedir / (str(cnt)+'.png'))
       cv.imwrite(file, frame)
       print("save cnt: ", cnt)

 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
