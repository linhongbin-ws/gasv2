
from csrk.arm_proxy import ArmProxy
from csrk.node import Node
from gym_ras.tool.csr_tool import T2CsrFrame, CsrFrame2T
from gym_ras.tool.common import getT, TxT, T2Euler
import numpy as np
from copy import deepcopy
import time
node_ = Node("NDDS_QOS_PROFILES.CSROS.xml") # NOTE: path Where you put the ndds xml file
psa3 = ArmProxy(node_, "psa3")
while(not psa3.is_connected):
    psa3.measured_cp()
    # To check if the arm is connected
    psa3.read_rtrk_arm_state()
    print("connection: ",psa3.is_connected)

# psa4 = ArmProxy(node_, "psa4")
print(psa3.measured_jp())

p = psa3.measured_cp()

world2base_yaw = 48

pose = CsrFrame2T(psa3.measured_cp())
pos, rot = T2Euler(pose)
print("current pose:", pos, rot)

T1 = getT(pos, [0, 0, world2base_yaw], euler_Degrees=True, rot_type="euler")
T2 = getT(pos, [-180, 0, 0], euler_Degrees=True, rot_type="euler")
T = TxT([T1, T2])
T[0:3,3] = pos
duration = 3
psa3.move_cp(T2CsrFrame(T), acc=1, duration=duration, jaw=1)
time.sleep(duration) # non blocking move
pos, rot = T2Euler(pose)
print("current pose:", pos, rot)

jp = psa3.measured_jp()

for i in range(6):
    if i!=2:
        value = np.rad2deg(jp[i])
    else:
        value = jp[i]
    print("- ", value)