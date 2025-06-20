import crtk, dvrk
import numpy as np
import time
ral = crtk.ral('dvrk_python_node')
_psm = dvrk.psm(ral, "PSM1")
ral.check_connections()
ral.spin()
_psm.jaw.move_jp(np.deg2rad(np.array([40]))).wait()
_psm.jaw.servo_jf(np.array([-0]))
time.sleep(2)
_psm.jaw.close().wait()
_psm.jaw.servo_jf(np.array([-0.16]))
time.sleep(5)
_psm.jaw.move_jp(np.deg2rad(np.array([20])))
# _psm.jaw.servo_jf(np.array([-0])).wait()
