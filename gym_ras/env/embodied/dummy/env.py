from gym_ras.env.embodied.base_env import BaseEnv
import gym
import numpy as np
from gym_ras.tool.common import getT

class DummyEnv(BaseEnv):
    def __init__(self,
                 robot_type,
                 **args):
        self.robot_type = robot_type
        client = None
        super().__init__(client)
        self._seed  = 0
        self._mask_id = {"psm1":1,"stuff":2}
        self._delta_transl = 0.05


    def get_oracle_action(self):
        pass

    def render(self):
        imgs ={}
        imgs['points'] = self._points.copy()
        imgs['rgb'] = np.zeros((600,600,3),np.uint8)
        imgs['mask'] = {'psm1': np.zeros((600,600),bool),
                        'stuff': np.zeros((600,600),bool),}
        return imgs
    def reset(self):
        obs = {"gripper_state": 1}
        return obs

    def reward_dict(self):
        pass
    def step(self, action):
        _transform_dict = {}
        # print(action)
        _transform_dict['psm1'] = getT(action[:3]*self._delta_transl, [0,0,0],rot_type="euler") 
        _transform_dict['stuff'] = getT([0,0,0], [0,0,0],rot_type="euler") 

        for k, v in _transform_dict.items():
            pc_idx = self._points[:, 6] == self._mask_id[k]
            ones = np.ones((self._points[pc_idx, :].shape[0], 1))
            P = np.concatenate((self._points[pc_idx, :3], ones), axis=1)
            self._points[pc_idx, :3] = np.matmul(P, np.transpose(v))[:, :3]

        obs = {"gripper_state": 1}
        done = False
        reward = 0 
        info = {'fsm': "prog_norm"}
        return obs, reward, done,  info

    def set_current_points(self, points):
        self._points = points.copy()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed


    @property
    def instrinsic_K(self):
        return None
    @property
    def action_space(self,):
        return gym.spaces.Box(low=-np.ones(5), high=np.ones(5))
    @property
    def observation_space(self):
        obs = {}
        obs['gripper_state'] = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        return gym.spaces.Dict(obs)