from gym_ras.api import make_env
from gym_ras.tool.img_tool import CV2_Visualizer
from gym_ras.tool.keyboard import Keyboard
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vis-tag', type=str, nargs='+', default=['rgb','depth','dsa','mask'])
args = parser.parse_args()
env, env_config = make_env(tags=["domain_random_enhance","dsa_occup2","gasv2_dvrk"], seed=0)
kb = Keyboard(blocking=False)
visualizer = CV2_Visualizer(
    update_hz=50,  vis_tag=args.vis_tag, keyboard=False
)

is_quit = False
ch = ''
while (ch!='q'):
    img = env.render()
    is_quit = visualizer.cv_show(img)
    ch = kb.get_char()
print("exit")
env.unwrapped.client._cam_device._device.close()

