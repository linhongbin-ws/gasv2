from gym_ras.env.wrapper.depth_process import DepthProcess
from gym_ras.env.wrapper.occup import Occup
import cv2
from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
import numpy as np
from gym_ras.tool.config import Config, load_yaml
from pathlib import Path
from gym_ras.env.wrapper import Visualizer

class env_i():
    def render(self):
        d2=np.load("render2.npy", allow_pickle=True)
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
env = Visualizer(env,                 vis_tag=[],
                 keyboard=True,)
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


