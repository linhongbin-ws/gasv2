from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import gym
import cv2


class ObsImage(BaseWrapper):
    def __init__(self, env,
                 new_image_map_key="dsa",
                 new_image_key="image",
                 new_image_size=64,
                 cv_interpolate="area",
                 **kwargs,
                 ):
        super().__init__(env,)
        self._new_image_map_key=new_image_map_key
        self._new_image_size=new_image_size
        self._new_image_key=new_image_key
        self._cv_interpolate = cv_interpolate
        self._obs_image_current = None


    def reset(self,):
        obs = self.env.reset()
        imgs = self._get_render_image()
        obs.update(imgs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        imgs = self._get_render_image()
        obs.update(imgs)
        return obs, reward, done, info

    def render(self, ):
        imgs = self._get_render_image()
        return imgs

    def _get_render_image(self):
        imgs = self.env.render()
        if self._new_image_map_key in imgs:
            imgs['image'] = self._resize(imgs[self._new_image_map_key], self._new_image_size)
        self._obs_image_current = imgs
        return imgs
    
    def _resize(self, img, size):
        return cv2.resize(img,
                         (size, size),
                        interpolation={"nearest": cv2.INTER_NEAREST,
                                        "linear": cv2.INTER_LINEAR,
                                        "area": cv2.INTER_AREA,
                        "cubic": cv2.INTER_CUBIC, }[self._cv_interpolate])
    
    @property
    def observation_space(self):
        obs = self.env.observation_space
        new_obs = {k: v for k,v in obs.items()}
        new_obs['image'] = gym.spaces.Box(0, 255, (self._new_image_size,self._new_image_size,3),
                                    dtype=np.uint8)

        return gym.spaces.Dict(new_obs)
    
    @property
    def obs_image_current(self):
        return self._obs_image_current
