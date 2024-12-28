from gym_ras.env.wrapper.base import BaseWrapper
from gym_ras.tool.common import scale_arr
import numpy as np
import gym
import cv2


class ObsVector(BaseWrapper):
    def __init__(self, env,
                new_vector_map_key=["gripper_state", "fsm_state",],
                vector2image_type="row",
                 **kwargs,
                 ):
        super().__init__(env,)
        self._new_vector_map_key = new_vector_map_key
        self._image_current = None
        self._vector2image_type = vector2image_type


    def reset(self,):
        obs = self.env.reset()
        new_obs = self._get_vector_obs(obs)
        obs.update(new_obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        new_obs = self._get_vector_obs(obs)
        obs.update(new_obs)
        return obs, reward, done, info

    def render(self, ):
        imgs = self.env.render()
        imgs['image'] = self._image_current.copy()
        return imgs

    def _get_vector_obs(self, obs):
        new_obs = {}
        obs_vector = obs[self._new_vector_map_key[0]] \
            if len(self._new_vector_map_key) == 1 \
            else np.concatenate(tuple([self._force_np(obs[k]) for k in self._new_vector_map_key]), axis=0)
        _low, _high = self._get_low_high_from_observation_space()
        new_obs['vector'] = scale_arr(obs_vector, _low, _high, -np.ones(_low.shape), np.ones(_low.shape))
        new_obs["image"] = self._vector2image(obs["image"], new_obs['vector'])
        self._image_current = new_obs["image"].copy()
        return new_obs
    
    def _force_np(self, x): 
        return x if isinstance(x, np.ndarray) else np.array([x])

    def _get_low_high_from_observation_space(self):
        low_high_tuple = lambda x: (x.low, x.high, x.shape)
        get_bound = lambda x, shape: x if x.shape[0] == shape else x*np.ones(shape)
        _list = [low_high_tuple(self.env.observation_space[k]) for k in self._new_vector_map_key]
        if len(_list) == 1:
            k = _list[0]
            _low = get_bound(k[0], k[2])
            _high = get_bound(k[1], k[2])
        else:
            _low = np.concatenate(
                tuple([get_bound(k[0], k[2]) for k in _list]), axis=0)
            _high = np.concatenate(
                tuple([get_bound(k[1], k[2]) for k in _list]), axis=0)
        return self._force_np(_low), self._force_np(_high)
    
    @property
    def observation_space(self):
        obs = self.env.observation_space
        dim = 0
        for v in self._new_vector_map_key:
            dim+=obs[v].shape[0]
        new_obs = {k: v for k,v in obs.items()}
        new_obs['vector'] = gym.spaces.Box(-1, 1, (dim, ), dtype=np.float32)
        return gym.spaces.Dict(new_obs)
    


    def _vector2image(self, image_in, vector, fill_channel=0, ):
        image = np.copy(image_in)
        _value_norm = np.clip(scale_arr(
            vector, old_min=-1, old_max=1, new_min=0.0, new_max=255.0), 0, 255.0)
        _value_norm = _value_norm.astype(np.uint8)
        # print("jj",_value_norm)
        if self._vector2image_type == "row":
            # print(_value_norm.shape)
            extend_d = image_in.shape[0] // vector.shape[0]
            _value_norm_l = np.zeros((_value_norm.shape[0] * extend_d,), dtype=np.uint8)
            # print(_value_norm)
            for i in range(_value_norm.shape[0]):
                _value_norm_l[i * extend_d: (i + 1) * extend_d] = _value_norm[i]
            # print(_value_norm_l)
            _value_norm = _value_norm_l
            _value_norm = np.tile(_value_norm, (image.shape[0], 1))
            _value_norm = np.transpose(_value_norm)
            s = image.shape[0]
            image[s-_value_norm.shape[0]:, :, fill_channel] = _value_norm
        elif self._vector2image_type == "square":
            ROW_SIZE = 6
            for _v in range(_value_norm.shape[0]):
                s = image.shape[0]
                a1 = s-_v*ROW_SIZE-1-ROW_SIZE
                b1 = s-_v*ROW_SIZE-1
                a2 = s-0*ROW_SIZE-1-ROW_SIZE
                b2 = s-0*ROW_SIZE-1
                image[a2:b2, a1:b1, fill_channel] = _value_norm[_v]


        elif self._vector2image_type == "pixel":
            _shape = image.shape
            _x = np.reshape(image[:, :, fill_channel], (-1))
            _x[_x.shape[0]-_value_norm.shape[0]:_x.shape[0]+1] = _value_norm
            _x = np.reshape(_x, image.shape[:2])
            image[:, :, fill_channel] = _x[:, :]
        # print(_value_norm)
        return image