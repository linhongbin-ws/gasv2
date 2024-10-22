import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from gym_ras.tool.rectify import my_rectify


cap_0=cv2.VideoCapture("/dev/video2")
cap_0.set(cv2.CAP_PROP_FPS, 60)

cap_0.set(3,1280)
cap_0.set(4,720)

cap_1=cv2.VideoCapture("/dev/video0")
cap_1.set(cv2.CAP_PROP_FPS, 60)

cap_1.set(3,1280)
cap_1.set(4,720)

counter = 0

while True:    
    ret1, frame1 = cap_0.read()  
    ret2, frame2 = cap_1.read()    
 
    frame1 = frame1[:,160:1120]
    frame1 = frame1[::-1]
    frame2 = frame2[:,160:1120]
    frame2 = frame2[::-1]

    frame1=cv2.resize(frame1, (800, 600))
    frame2=cv2.resize(frame2, (800, 600))

    cv2.imshow("Input1", frame1)   
    cv2.imshow("Input2", frame2)    

    if cv2.waitKey(1) == ord('a'):
        counter+=1
        cv2.imwrite('./data/cam_cal/l/' + str(counter) + '.png', frame1)
        cv2.imwrite('./data/cam_cal/r/' + str(counter)+ '.png', frame2)    
    elif cv2.waitKey(1) == ord('q'):
        break
cap_0.release()
cap_1.release()
cv2.distroyAllWindows()