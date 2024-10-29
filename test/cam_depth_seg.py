from gym_ras.api import make_env
from gym_ras.tool.img_tool import CV2_Visualizer
from gym_ras.tool.keyboard import Keyboard
import numpy as np
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
args = parser.parse_args()
# env, env_config = make_env(tags=['gasv2_dvrk'], seed=0)
# kb = Keyboard(blocking=False)
# visualizer = CV2_Visualizer(
#     update_hz=50,  vis_tag=args.vis_tag, keyboard=False
# )



d2=np.load("render.npy", allow_pickle=True)
d2.item().get('key2')
keys=['rgb', 'depReal', 'mask']
img = {k: d2.item().get(k) for k in keys}
def erode_mask(mask, kernel_size=7):
    in_mask = np.zeros(mask.shape, dtype=np.uint8)
    in_mask[v] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    erode = cv2.morphologyEx(in_mask, cv2.MORPH_ERODE, kernel)
    erode_mask = erode !=0
    return erode_mask

# img = env.render()

from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt

seg_depth = []

for k,v in img['mask'].items():
    t = img['depReal'].copy()
    t[np.logical_not(v)] = 0
    seg_depth.append(t)

    t1 = img['depReal'].copy()
    v1 = erode_mask(v)
    t1[np.logical_not(v1)] = 0
    seg_depth.append(t1)

plot_list = [img['rgb'], img['depReal']]
plot_list.extend(seg_depth)

plot_list = [v for i, v in enumerate(plot_list) if i in [2,3]]

for n, v in enumerate(plot_list):
    plt.subplot(1,len(plot_list), n+1)
    imshow(v)
    plt.colorbar()
show()


print("exit")
# env.unwrapped.client._cam_device._device.close()



