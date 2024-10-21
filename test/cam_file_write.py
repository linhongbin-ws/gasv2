import cv2
import numpy as np
fs = cv2.FileStorage("./data/cam_cal.yaml", cv2.FILE_STORAGE_WRITE)

M1 = np.array([
    [379.896139278856, 0.0, 308.8882016858996,],
[0.0, 506.4881668483393, 216.38516077848874,] ,
[0.0, 0.0, 1.0 ,],
]) 


fs.write('M1', M1)