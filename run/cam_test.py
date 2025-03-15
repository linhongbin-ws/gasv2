import numpy as np
import cv2 as cv
 
cap = cv.VideoCapture('/dev/video2')
cap.set(cv.CAP_PROP_FPS, 60)
cap.set(3,1280)
cap.set(4,720)
if not cap.isOpened():
 print("Cannot open camera")
 exit()
while True:
 # Capture frame-by-frame
    ret, frame = cap.read()

    frame = frame[:,160:1120]
    frame = frame[::-1]
    frame=cv.resize(frame, (800, 600))
    frame = frame[:,100:700,:]
    # print(frame.shape)
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2RG)
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
print(frame.shape)