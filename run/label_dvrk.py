from gym_ras.tool.track_any_tool import Tracker
from gym_ras.api import make_env
from gym_ras.env.wrapper import Visualizer, ActionOracle
import argparse
from tqdm import tqdm
import time
env, env_config = make_env(tags=['gasv2_dvrk'], seed=0)
print("start tracker....")
tracker = Tracker(sam=True, xmem=False)
img = env.client._cam_device._device.get_image()
tracker.set_rgb_image(img['rgb'])
tracker.gui()