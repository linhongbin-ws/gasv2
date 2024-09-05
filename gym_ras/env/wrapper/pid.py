from gym_ras.env.wrapper.base import BaseWrapper
from gym_ras.tool.seg_tool import get_mask_boundary
import numpy as np
import gym

def mean(x,y):
    return np.mean(np.array([x,y]))
class PID(BaseWrapper):
    def __init__(
        self, env, 
         obs_type='occup',
         control_p=10,
         phase_thres = [0.1,0.1,0.05],
         err_offset = [0,0,0.15],
           **kwargs
    ):
        super().__init__(env, **kwargs)
        self._obs_type = obs_type
        self._control_p = control_p
        self._phase_thres  =phase_thres
        self._err_offset = err_offset
    
    def reset(self):
        obs = self.env.reset()
        # _obs = self._get_pid_observation()
        obs["controller_state"] = 1
        # obs["controller_state"] = 1  if self._check_pid_phase(_obs) else 0
        return obs

    def step(self, action):
        obs = self._get_pid_observation()
        pid_phase = False
        if self._check_pid_phase(obs):
            action = self._get_pid_action(obs)
            pid_phase =True 
            
        obs, reward, done, info = self.env.step(action)
        obs["controller_state"] = 1  if pid_phase else 0
        return obs, reward, done, info

    def _get_pid_observation(self):
        if self._obs_type == 'occup':
            occ = self.env.occupancy_projection
            centroids = {}
            
            for k, v in occ.items():
                x = self._get_mask_centroid(v[0])
                y = self._get_mask_centroid(v[1])
                z = self._get_mask_centroid(v[2])
                centroids[k] = [x,y,z]
            x1 = centroids['psm1'][0][0]-centroids['stuff'][0][0]
            x2 = centroids['psm1'][2][0]-centroids['stuff'][2][0]
            y1 = -(centroids['psm1'][1][0]-centroids['stuff'][1][0])
            y2 = centroids['psm1'][2][1]-centroids['stuff'][2][1]
            z1 = centroids['psm1'][0][1]-centroids['stuff'][0][1]
            z2 = centroids['psm1'][1][1]-centroids['stuff'][1][1]

            x_err = mean(x1, x2) 
            y_err = mean(y1, y2)
            z_err = mean(z1, z2)
            print("x", x1, x2, x_err)
            print("y", y1, y2, y_err)
            print("z", z1, z2, z_err)
            obs = {}
            obs['err'] = [ x_err, y_err, z_err]
        return  obs
    
    def _check_pid_phase(self, obs):
        err1 = np.array(obs['err'])
        err2 = err1 + np.array(self._err_offset)
        err = np.minimum(np.abs(err1),np.abs(err2))
        phase = lambda d: np.abs(err[d])>self._phase_thres[d] and (err1[d] * err2[d] > 0)
        x_phase = phase(0)
        y_phase = phase(1)
        z_phase = phase(2)
        print("phase", x_phase,y_phase,z_phase)
        return x_phase or y_phase or z_phase
        
    def _get_pid_action(self, obs):
            x_err, y_err, z_err = obs['err'][0], obs['err'][1], obs['err'][2]
            action = np.array([-self._control_p*(y_err+self._err_offset[1]),
                                -self._control_p*(x_err+self._err_offset[0]),
                                self._control_p*(z_err+self._err_offset[2]),
                     0,
                     0,])
            # action = np.array([0,0,2,0,0])
            action = np.clip(action, -np.ones(action.shape), np.ones(action.shape))
            print("action is ", action)
            return action

    def _get_mask_centroid(self, mask):
        _, _, r, c =get_mask_boundary(mask)
        return [mean(c[0],c[1]), mean(r[0],r[1])]

    @property
    def observation_space(self):
        obs = {k: v for k, v in self.env.observation_space.items()}
        obs['controller_state'] = gym.spaces.Box(low=0,
                                          high=1, shape=(1,), dtype=np.float32)
        return gym.spaces.Dict(obs)