import PyCSR
import numpy as np

def T2CsrFrame(T):
    pos = PyCSR.Vector(T[0,3],T[1,3],T[2,3])
    rot=PyCSR.Rotation(
        T[0,0],T[0,1],T[0,2],
        T[1,0],T[1,1],T[1,2],
        T[2,0],T[2,1],T[2,2],
    )
    frame = PyCSR.Frame(rot, pos)
    return frame

def CsrFrame2T(frame):
    m = frame.M
    T = np.array(
        [
            [m[0, 0], m[0, 1], m[0, 2], frame.p[0]],
            [m[1, 0], m[1, 1], m[1, 2], frame.p[1]],
            [m[2, 0], m[2, 1], m[2, 2], frame.p[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return T