import sys
from gym_ras.api import make_env
from gym_ras.tool.depth import get_intrinsic_matrix, projection_matrix_to_K
from gym_ras.tool.sym import generate_sym3, get_sym_params,a2T_discrete, discrete_action_inverse, aggregate_Ts
from gym_ras.tool.plt import use_backend, plot_img, plot_traj
import numpy as np

tags = ['domain_random_enhance', 'dsa_occup2','sym','no_clutch', 'no_pid','no_action_noise']
env, env_config = make_env(tags=tags)
obs = env.reset()
obss_origin = []
steps_num = 6
origin_a = 2
reverse_a =3
obss_origin.append(obs)
actions_origin = []
for i in range(steps_num):
    action = origin_a
    obs,reward, done, info = env.step(action)
    obss_origin.append(obs)
    actions_origin.append(action)

print(obs.keys())
dummy_env, env_config = make_env(tags=tags +['dummy',])
# dummy_env.set_current_points(obs['points'])
new_sym_obss, new_sym_actionss = generate_sym3(obss_origin,actions_origin, sym_start_step=3,dummy_env=dummy_env)


imgss = []
imgss.append([{"image": np.minimum(v['occup_zimage']['psm1'][0],v['occup_zimage']['stuff'][0]), "title": f"GT step {i}"} for i, v in enumerate(obss_origin)])
for _new_obs in new_sym_obss:
    imgss.append([{"image": np.minimum(v['occup_zimage']['psm1'][0],v['occup_zimage']['stuff'][0]), "title": f"Sym step {i}"} for i, v in enumerate(_new_obs)])

plot_img(imgss)





points_mats = []
trajTs = [a2T_discrete(discrete_action_inverse(a), 0.03) for a in reversed(actions_origin)]
trajTs = aggregate_Ts(trajTs)
points_mats = []
pc_mat = {}
pc_mat["mat"] = np.array([[t[0][3], t[1][3], t[2][3]] for t in trajTs])
pc_mat["linewidth"] = 4
pc_mat["alpha"] = 1
pc_mat["label"] = f"original traj"
pc_mat["color"] = "k"
points_mats.append(pc_mat)

for i, new_actions in enumerate(new_sym_actionss):
    new_trajTs = [a2T_discrete(discrete_action_inverse(a), 0.03) for a in reversed(new_actions)]
    new_trajTs = aggregate_Ts(new_trajTs)
    pc_mat = {}
    pc_mat["mat"] = np.array([[t[0][3], t[1][3], t[2][3]] for t in new_trajTs])
    pc_mat["linewidth"] = 4
    pc_mat["alpha"] = 0.5
    pc_mat["label"] = f"traj {i+1}"
    points_mats.append(pc_mat)   
plot_traj(points_mats, elev=45, azim=135, roll=0)


