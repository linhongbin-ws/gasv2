import sys
from gym_ras.api import make_env
from gym_ras.tool.depth import get_intrinsic_matrix, projection_matrix_to_K
from gym_ras.tool.sym import generate_sym2, get_sym_params
from gym_ras.tool.plt import use_backend, plot_img
import numpy as np
env, env_config = make_env(tags=['domain_random_enhance', 'dsa_occup2','sym','no_clutch'])
obs = env.reset()
proj_matrix = env.get_intrinsic_matrix
K = projection_matrix_to_K(proj_matrix, image_size=600)
# print("K", K)

origin_actions = []
obss_origin = [obs]
done = False
for i in range(4):
    action = env.get_oracle_action()
    origin_actions.append(env.action_dis2con(action))
    obs, reward, done, info = env.step(action)
    obss_origin.append(obs)
print(obs.keys())
# print(obs['occup_zimage'])
# print(obs['occup_zimage'].shape)



imgss = []
imgss.append([{"image": np.minimum(v['occup_zimage']['psm1'][0],v['occup_zimage']['stuff'][0]), "title": f"GT step {i}"} for i, v in enumerate(obss_origin)])
# for new_obs in new_obss:
#     imgss.append([{"image": v['image'][0,:,:], "title": f"Aug step {i}"} for i, v in enumerate(new_obs)])
# plt.rcParams['figure.figsize'] = [30, 10]
plot_img(imgss)