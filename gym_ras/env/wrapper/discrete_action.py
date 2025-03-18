from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import gym


class DiscreteAction(BaseWrapper):
    def __init__(self, env,
                 pos_action_scale=0.2,
                 rot_action_scale=1,
                 is_discrete=True,
                 **kwargs):
        super().__init__(env)
        self._action_dim = self.env.action_space.shape[0]
        assert self._action_dim == 5  # onlys support x,y,z,yaw,gripper
        self._action_list = []
        self._action_strs = ['x_neg', 'x_pos', 'y_neg', 'y_pos',
                             'z_neg', 'z_pos', 'rot_neg', 'rot_pos', 'gripper_toggle']
        self._action_prim = {}  # store discrete action primitives
        self._action_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self._action_discrete_n = len(self._action_idx)
        self._is_discrete = is_discrete
        self._pos_action_scale = pos_action_scale
        self._rot_action_scale = rot_action_scale

        # create map
        for i in range(4):
            _action = np.zeros(self._action_dim)
            _action[i] = -pos_action_scale if i!=3 else -rot_action_scale
            self._action_prim[self._action_strs[2*i]] = _action
            _action = np.zeros(self._action_dim)
            _action[i] = pos_action_scale if i!=3 else rot_action_scale
            self._action_prim[self._action_strs[2*i+1]] = _action
        self._action_prim['gripper_toggle'] = np.zeros(self._action_dim)

    @property
    def action_space(self):
        if self._is_discrete:
            return gym.spaces.Discrete(self._action_discrete_n)
        else:
            return self.env.action_space

    def reset(self):
        obs = self.env.reset()
        self._reset_vars()
        return obs

    def _reset_vars(self):
        self._is_gripper_close = False
    
    def action_dis2con(self, action):
        return self._action_prim[self._action_strs[self._action_idx[action]]]


    def step(self, action):
        # print(action)
        if self._is_discrete:
            _action = self._action_prim[self._action_strs[self._action_idx[action]]]
            if self._action_strs[self._action_idx[action]] == 'gripper_toggle':
                self._is_gripper_close = not self._is_gripper_close

            if self._is_gripper_close:
                _action[-1] = -1
            else:
                _action[-1] = 1
        else:
            _action = action.copy()
            _action[:3] *= self._pos_action_scale  
            _action[3] *= self._rot_action_scale     
        # print(_action)
        # print(_action, self._is_gripper_close)
        # print("input action", action, _action, self._action_idx[action], self._action_strs[self._action_idx[action]],self._action_prim[self._action_strs[self._action_idx[action]]])
        obs, reward, done, info = self.env.step(_action)

        return obs, reward, done, info

    def get_oracle_action(self):
        if self._is_discrete:
            action = self.env.get_oracle_action()
            
            ref = np.zeros(action.shape)
            if self._is_gripper_close:
                ref[-1] = -1
            else:
                ref[-1] = 1

            _err = action - ref
            # print("err",_err, self._is_gripper_close)
            if np.abs(_err)[-1] >= 1:  # gripper toggle
                _action = 8
                return _action
            else:
                _err[-1] = 0
                _err[-2] *= 0.4 # reduce rotation weight
                _index = np.argmax(np.abs(_err))
                _direction = _err[_index] > 0
                if _direction:
                    _action = _index * 2 + 1
                else:
                    _action = _index * 2
            return _action
        else:
            _action = self.env.get_oracle_action()
            _action[:3] /= self._pos_action_scale  
            _action[3] /= self._rot_action_scale
            _action = np.clip(_action, -1, 1)           
            return _action