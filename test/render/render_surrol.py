from gym_ras.api import make_env
from gym_ras.env.wrapper import Visualizer, ActionOracle
from gym_ras.tool.plt import plot_img, use_backend, get_backend

show_steps = 6

env, env_config = make_env(tags=['gasv2_surrol', 'no_clutch', 'dsa2'], seed=0)
done = False
obs = env.reset()
obs_traj = [obs]
while not done:
    action = env.get_oracle_action()
    obs, reward, done, info = env.step(action)
    obs_traj.append(obs)
    
imgss = []
imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][:,:,0] for o in obs_traj[:show_steps]])])
imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][:,:,1] for o in obs_traj[:show_steps]])])
imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][:,:,2] for o in obs_traj[:show_steps]])])
# imgss.append(sym_depth_image_traj)
# imgss.append(sym_depth_image_traj2)
# imgss.append(gt_env_traj_with_sym_actions)
use_backend('tkagg')
plot_img(imgss)
print("backend:", get_backend())