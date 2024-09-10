from gym_ras.tool.track_any_tool import Tracker


from gym_ras.api import make_env
from gym_ras.env.wrapper import Visualizer, ActionOracle
import argparse
from tqdm import tqdm
import time
parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-p',type=int)
parser.add_argument('--repeat',type=int, default=1)
parser.add_argument('--action',type=str, default="oracle")
# parser.add_argument('--yaml-dir', type=str, default="./gym_ras/config.yaml")
# parser.add_argument('--yaml-tag', type=str, nargs='+', default=[])
parser.add_argument('--env-tag', type=str, nargs='+', default=['gas_surrol','csr_grasp_any'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
parser.add_argument('--oracle', type=str, default='keyboard')
parser.add_argument('--no-vis', action="store_true")
parser.add_argument('--eval', action="store_true")

args = parser.parse_args()

env, env_config = make_env(tags=args.env_tag, seed=args.seed)
env.unwrapped.client.reset_pose()
env.unwrapped.client._cam_device._device._depth_remap = False

# print(img['rgb'].shape)


print("start tracker....")
tracker = Tracker(sam=True, xmem=False)
img = env.render()
tracker.set_rgb_image(img['rgb'])

tracker.gui()
# tracker.track(img['rgb'])
# env =  Visualizer(env, update_hz=10, keyboard=True)
# env.unwrapped.client._cam_device._segment.model = tracker
# while True:
#     img = env.render()
#     # img['mask'] = {}
#     # mask = tracker.track(img['rgb'])

#     # img['mask']['stuff'] = mask == 1
#     # img['mask']['psm1'] = mask == 2
#     # img['mask']['psm1_except_gripper'] = mask == 3
#     if len(args.vis_tag) != 0:
#         img = {k:v for k,v in img.items() if k in args.vis_tag}
    
#     img_break = env.cv_show(imgs=img)
#     if img_break:
#         break