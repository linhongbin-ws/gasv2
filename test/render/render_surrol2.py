import open3d as o3dd
# o3dd.visualization.draw_geometries([])
from gym_ras.api import make_env
# o3dd.visualization.draw_geometries([])
# from gym_ras.env.wrapper import Visualizer, ActionOracle
# from gym_ras.tool.plt import plot_img, use_backend, get_backend
from gym_ras.tool.o3d import convert_occup_mat_o3dvoxels, O3DVis
# o3dd.visualization.draw_geometries([])
# import matplotlib
import time 


show_steps = 6

env, env_config = make_env(tags=['gasv2_surrol', 'no_clutch', 'dsa2'], seed=0)
# o3dd.visualization.draw_geometries([])
done = False
obs = env.reset()
imgs  = env.render()
vis = O3DVis()
# print(imgs['occup_mat'])
for i in range(2):
    vx_grids = convert_occup_mat_o3dvoxels(
        occ_mat_dict=imgs['occup_mat'],
        pc_x_min=env_config.wrapper.Occup.pc_x_min,
        pc_x_max=env_config.wrapper.Occup.pc_x_min,
        pc_y_min=env_config.wrapper.Occup.pc_y_min,
        pc_y_max=env_config.wrapper.Occup.pc_y_max,
        pc_z_min=env_config.wrapper.Occup.pc_z_min,
        pc_z_max=env_config.wrapper.Occup.pc_z_max,
        voxel_size=env.occup_grid_size
    )
    vis.draw([v for _, v in vx_grids.items()])
    time.sleep(1)
# obs_traj = [obs]
# while not done:
#     action = env.get_oracle_action()
#     obs, reward, done, info = env.step(action)
#     obs_traj.append(obs)
    
# imgss = []
# imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][:,:,0] for o in obs_traj[:show_steps]])])
# imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][:,:,1] for o in obs_traj[:show_steps]])])
# imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][:,:,2] for o in obs_traj[:show_steps]])])
# # imgss.append(sym_depth_image_traj)
# # imgss.append(sym_depth_image_traj2)
# # imgss.append(gt_env_traj_with_sym_actions)
# use_backend('tkagg')
# plot_img(imgss)
# print("backend:", get_backend())