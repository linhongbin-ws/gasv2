from gym_ras.api import make_env
from gym_ras.tool.img_tool import CV2_Visualizer
from gym_ras.tool.keyboard import Keyboard
import numpy as np
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
parser.add_argument('--savedir', default="./render.npy")
args = parser.parse_args()
env, env_config = make_env(tags=['gasv2_dvrk', 'no_depth_process'], seed=0)





img = env.render()
np.save(args.savedir, img)


print("exit")
env.unwrapped.client._cam_device._device.close()



