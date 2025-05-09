from gym_ras.env.wrapper.depth_process import DepthProcess
from gym_ras.env.wrapper.occup import Occup
from gym_ras.env.wrapper.dsa import DSA
import cv2
from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
import numpy as np
from gym_ras.tool.config import Config, load_yaml
from pathlib import Path
from gym_ras.env.wrapper import Visualizer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, default='./render3.npy')
parser.add_argument('--vis-tag',nargs='+', type=str, default=[])
args = parser.parse_args()


class env_i():
    def __init__(self) -> None:
        self.timestep =0
    def render(self):
        d2=np.load(args.indir, allow_pickle=True)
        keys=['rgb', 'depReal', 'mask']
        img = {k: d2.item().get(k) for k in keys}
        return img




yaml_dir = Path("./gym_ras/config/gym_ras.yaml")
yaml_dict = load_yaml(yaml_dir)
yaml_config = yaml_dict["default"].copy()
config = Config(yaml_config)
config = config.update(yaml_dict['gasv2_dvrk'])

env = env_i()
env = DepthProcess(env,**getattr(config.wrapper, "DepthProcess").flat)
env = Occup(env,**getattr(config.wrapper, "Occup").flat)
env = DSA(env,**getattr(config.wrapper, "DSA").flat)
env = Visualizer(env,                
                 keyboard=True,
                 vis_tag=args.vis_tag)
imgs = env.render()
img_break = env.cv_show(imgs=imgs)


# env = env_i()
# env = DepthProcess(env,**getattr(config.wrapper, "DepthProcess").flat)
# env._edge_detect_thres = 0
# env = Occup(env,**getattr(config.wrapper, "Occup").flat)
# env = Visualizer(env,                 vis_tag=[],
#                  keyboard=True,)
# imgs = env.render()
# img_break = env.cv_show(imgs=imgs)

# env = env_i()
# env = DepthProcess(env,**getattr(config.wrapper, "DepthProcess").flat)
# env._edge_detect_thres = 0
# env._erode_kernel = 8
# env = Occup(env,**getattr(config.wrapper, "Occup").flat)
# env = Visualizer(env,                 vis_tag=[],
#                  keyboard=True,)
# imgs = env.render()
# img_break = env.cv_show(imgs=imgs)


