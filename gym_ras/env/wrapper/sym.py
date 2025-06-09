from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import gym
from gym_ras.tool.sym import generate_sym3
from copy import deepcopy as cp

class Sym(BaseWrapper):


    def __init__(self, env,
                 dummy_env,
                 sym_action_noise=0.3,
                 sym_aug_new_eps=12,
                 keep_sym_obs_key = None,
                 **kwargs,
                 ):
        super().__init__(env)
        self._dummy_env = dummy_env
        self._sym_eps = []
        self._eps_buffer = []
        self._is_sym = True
        self._step = 0
        self._sym_action_noise = sym_action_noise
        self._sym_aug_new_eps = sym_aug_new_eps
        self._keep_sym_obs_key = keep_sym_obs_key

    def reset(self,):
        self._step = 0
        if len(self._sym_eps) != 0 and self._is_sym:
            obs,reward, done, info, action = self._sym_eps[0][self._step]
            obs['sym_state'] = 1
            return obs
        else:
            obs = self.env.reset()
            reward, done, info, action = None, None, None, 0
            if self._is_sym:
                self._eps_buffer = []
                self._eps_buffer.append((cp(obs), cp(reward), cp(done), cp(info), cp(action)))
            obs['sym_action'] = action
            obs['sym_state'] = 0
            return obs



    def step(self, action):
        self._step += 1
        if len(self._sym_eps) != 0 and self._is_sym:
            obs,reward, done, info, action = self._sym_eps[0][self._step]
            obs['sym_state'] = 1
            if done:
                self._sym_eps = self._sym_eps[1:]
        else:
            obs, reward, done, info = self.env.step(action)
            obs['sym_action'] = action
            obs['sym_state'] = 0
            if self._is_sym:
                self._eps_buffer.append((cp(obs), cp(reward), cp(done), cp(info), cp(action)))
                if done:
                    self._on_end_gt_eps()
            return obs, reward, done, info


        return obs, reward, done, info

    
    def _on_end_gt_eps(self,):
        obss_origin = [v[0] for v in self._eps_buffer]
        actions_origin = [v[4] for v in self._eps_buffer]
        actions_origin = actions_origin[1:]

        new_sym_obss = []
        new_sym_actionss = []
        for _ in range(self._sym_aug_new_eps):
            new_sym_obs, new_sym_actions = generate_sym3(obss_origin,actions_origin,
                                                         sym_action_noise = self._sym_action_noise,
                                                          dummy_env=self._dummy_env,
                                                       sym_start_step=0,)
            if self._keep_sym_obs_key is not None:
                _keep_key = self._keep_sym_obs_key 
                new_sym_obs = [ {_k: _o[_k] for _k in _keep_key} for _o in new_sym_obs]
            print("len new_sym_obss:", len(new_sym_obss))
            new_sym_obss.append(new_sym_obs)
            new_sym_actionss.append(new_sym_actions)


        for i in range(len(new_sym_obss)):
            _ep = []
            for j in range(len(self._eps_buffer)):
                obs = cp(new_sym_obss[i][j])
                reward = cp(self._eps_buffer[j][1])
                done = cp(self._eps_buffer[j][2])
                info = cp(self._eps_buffer[j][3])
                action = 0 if j == 0 else cp(new_sym_actionss[i][j-1])
                obs['sym_action']  = action
                _ep.append((obs, reward, done, info, action,))
            self._sym_eps.append(_ep)
            
        self._eps_buffer = []

    def set_sym(self, is_sym):
        self._is_sym = is_sym



    @property
    def observation_space(self):
        obs = {k: v for k, v in self.env.observation_space.items()}
        obs['sym_action'] = gym.spaces.Box(low=0,
                                          high=8, shape=(1,), dtype=float)
        obs['sym_state'] = gym.spaces.Box(low=0,
                                          high=1, shape=(1,), dtype=float)
        return gym.spaces.Dict(obs)
        

    def mea_rollouts(self, eps):
        self._sym_aug_new_eps = eps
    
    @property
    def sym_aug_new_eps(self):
        return self._sym_aug_new_eps