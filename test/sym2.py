import sys
from gym_ras.api import make_env
from gym_ras.tool.depth import get_intrinsic_matrix, projection_matrix_to_K
from gym_ras.tool.sym import generate_sym2, get_sym_params
from gym_ras.tool.plt import use_backend, plot_img
import numpy as np

tags = ['domain_random_enhance', 'dsa_occup2','sym','no_clutch', 'no_pid','no_action_noise']
env, env_config = make_env(tags=tags)
obs = env.reset()
obss_origin = []
obss_origin.append(obs)
for i in range(3):
    action = 2
    obs,reward, done, info = env.step(action)
    obss_origin.append(obs)

print(obs.keys())
dummy_env, env_config = make_env(tags=tags +['dummy',])
dummy_env.set_current_points(obs['points'])
obs = dummy_env.reset()
obss_new = []
obss_new.append(obs)
for i in range(3):
    action = 3
    obs,reward, done, info = dummy_env.step(action)
    obss_new.append(obs)
obss_new = [v for v in reversed(obss_new)]


imgss = []
imgss.append([{"image": np.minimum(v['occup_zimage']['psm1'][0],v['occup_zimage']['stuff'][0]), "title": f"GT step {i}"} for i, v in enumerate(obss_origin)])
imgss.append([{"image": np.minimum(v['occup_zimage']['psm1'][0],v['occup_zimage']['stuff'][0]), "title": f"Sym step {i}"} for i, v in enumerate(obss_new)])

plot_img(imgss)