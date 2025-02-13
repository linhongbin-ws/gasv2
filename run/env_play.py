from gym_ras.api import make_env
from gym_ras.env.wrapper import Visualizer, ActionOracle
import argparse
from tqdm import tqdm
import time
parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help')

parser.add_argument('-p', type=int)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--action', type=str, default="oracle")
# parser.add_argument('--yaml-dir', type=str, default="./gym_ras/config.yaml")
# parser.add_argument('--yaml-tag', type=str, nargs='+', default=[])
parser.add_argument('--env-tag', type=str, nargs='+', default=[])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
parser.add_argument('--oracle', type=str, default='keyboard')
parser.add_argument('--no-vis', action="store_true")
parser.add_argument('--eval', action="store_true")
parser.add_argument('--vis-occup', action="store_true")


args = parser.parse_args()

env, env_config = make_env(tags=args.env_tag, seed=args.seed)
print(env_config)
if args.action == 'oracle':
    env = ActionOracle(env, device=args.oracle)
if not args.no_vis:
    env = Visualizer(env, update_hz=100 if args.action in [
                     'oracle'] else -1, vis_tag=args.vis_tag, keyboard=not args.action in ['oracle'])

if args.vis_occup:
    from gym_ras.tool.o3d import convert_occup_mat_o3dvoxels, O3DVis
    x_c = (env_config.wrapper.Occup.pc_x_min + env_config.wrapper.Occup.pc_x_max) / 2
    y_c = (env_config.wrapper.Occup.pc_y_min + env_config.wrapper.Occup.pc_y_max) / 2
    z_c = (env_config.wrapper.Occup.pc_z_min + env_config.wrapper.Occup.pc_z_max) / 2
    vis_occup = O3DVis(center_list=[x_c, y_c, z_c])
if args.eval:
    env.to_eval()
print("action space: ", env.action_space)
print("observation space: ", env.observation_space)
eps_cnt =0 
eps_success_cnt=0
for _ in tqdm(range(args.repeat)):
    done = False
    obs = env.reset()
    if not args.no_vis:
        img = env.render()
        img_break = env.cv_show(imgs=img)
        # img_break = env.cv_show(imgs=img)
    # print("obs:", obs)
    while not done:
        # action = env.action_space.sample()
        # print(action)
        print("==========step", env.timestep, "===================")
        if any(i.isdigit() for i in args.action):
            action = int(args.action)
        elif args.action == "random":
            action = env.action_space.sample()
        elif args.action == "oracle":
            action = env.get_oracle_action()
            if action == 'quit':
                break
        else:
            raise NotImplementedError
        # print("step....")
        obs, reward, done, info = env.step(action)
        if done:
            eps_cnt +=1 
            if info['fsm'] == "done_success":
                eps_success_cnt+=1
            print("************")
            print()
            print(f"fsm state: {info['fsm']}, success/total: ({eps_success_cnt}/{eps_cnt})")
            print()
            print("************")
        print_obs = obs.copy()
        print_obs = {k: v.shape if hasattr(v, 'shape') else "" for k, v in print_obs.items()}
        print_obs = [str(k) + ":" + str(v) for k, v in print_obs.items()]
        print(" | ".join(print_obs))
        print("reward:", reward, "done:", done,)
        print("info:", info)

        # print("reward:", reward, "done:", done, "info:", info, "step:", env.timestep, "obs_key:", obs.keys(), "fsm_state:", obs["fsm_state"])
        # print("observation space: ", env.observation_space)
        start = time.time()
        # print(obs["image"])
        img = env.render()
        print("render() elapse sec: ", time.time() - start)
        # print(img.keys())
        # print(img)
        img.update({"image": obs['image']})
        if "dsa" in img:
            obs.update({"dsa": img["dsa"]})
        # # print(action)
        # if len(args.vis_tag) != 0:
        #     img = {k:v for k,v in img.items() if k in args.vis_tag}
            
        if args.vis_occup:
            vx_grids = convert_occup_mat_o3dvoxels(
                occ_mat_dict=img['occup_mat'],
                pc_x_min=env_config.wrapper.Occup.pc_x_min,
                pc_x_max=env_config.wrapper.Occup.pc_x_max,
                pc_y_min=env_config.wrapper.Occup.pc_y_min,
                pc_y_max=env_config.wrapper.Occup.pc_y_max,
                pc_z_min=env_config.wrapper.Occup.pc_z_min,
                pc_z_max=env_config.wrapper.Occup.pc_z_max,
                voxel_size=env.occup_grid_size
            )
            vis_occup.draw([v for _, v in vx_grids.items()])
        if not args.no_vis:
            img_break = env.cv_show(imgs=img)
            if img_break:
                break
    if img_break or  action == 'quit':
        break
    if not args.no_vis:
        if img_break:
            break

if "gasv2_dvrk" in args.env_tag:
    env.unwrapped.client._cam_device._device.close()
