from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import gym


class PID(BaseWrapper):
    def __init__(
        self, env,
        obs_type='occup',
        control_p=10,
        phase_thres=[0.1, 0.1, 0.05],
        err_offset=[0.0, 0.0, 0.10],
        skip=False,
        fsm_z_err_min=-0.02,
        fsm_z_err_state='prog_abnorm_1',
        **kwargs
    ):
        super().__init__(env, **kwargs)
        self._obs_type = obs_type
        self._control_p = control_p
        self._phase_thres = phase_thres
        self._err_offset = err_offset
        self._skip = skip
        self._fsm_z_err_min = fsm_z_err_min
        self._fsm_z_err_state = fsm_z_err_state

    def reset(self):
        obs = self.env.reset()
        obs["controller_state"] = 1
        return obs

    def step(self, action):
        pid_obs = self._get_pid_observation()
        pid_phase = False
        check_phase, x_phase, y_phase, z_phase = self._check_pid_phase(pid_obs)
        if check_phase and (not self._skip):
            action = self._get_pid_action(pid_obs, x_phase, y_phase, z_phase)
            pid_phase = True

        obs, reward, done, info = self.env.step(action)
        obs["controller_state"] = 1 if pid_phase else 0
        info["controller_state"] = obs["controller_state"]

        if pid_obs['err'][2] < self._fsm_z_err_min:
            info['fsm'] =  self._fsm_z_err_state
            print(f"exceed z err, err: {pid_obs['err'][2]}, thres: {self._fsm_z_err_min}")
        # print("pid state", obs["controller_state"])
        return obs, reward, done, info

    def _get_pid_observation(self):
        if self._obs_type == 'occup':
            occ = self.env.occup_mat
            centroids = {}

            for k, v in occ.items():
                xs, ys, zs = v.nonzero()
                gs = self.env.occup_grid_size
                centroids[k] = [
                    np.mean(xs)*gs, np.mean(ys)*gs, np.mean(zs)*gs]
                # print(f"{k} centroid: {centroids[k]}")
            x_err = centroids['psm1'][0]-centroids['stuff'][0] + self._err_offset[0]
            y_err = centroids['psm1'][1]-centroids['stuff'][1] + self._err_offset[1]
            z_err = centroids['psm1'][2]-centroids['stuff'][2] + self._err_offset[2]
            # print(f"Centroid error: {x_err} {y_err} {z_err}")
            obs = {}
            obs['err'] = [x_err, y_err, z_err]
        return obs

    def _check_pid_phase(self, obs):
        err = np.array(obs['err'])
        def phase(d): return np.abs(err[d]) > self._phase_thres[d] 
        x_phase = phase(0)
        y_phase = phase(1)
        z_phase = phase(2)
        # print("phase:", x_phase,y_phase,z_phase)
        return x_phase or y_phase or z_phase, x_phase, y_phase, z_phase

    def _get_pid_action(self, obs, x_phase, y_phase, z_phase):
        x_err, y_err, z_err = obs['err'][0], obs['err'][1], obs['err'][2]
        action = np.array([-self._control_p * (x_err+self._err_offset[0]) if x_phase else 0,
                           -self._control_p *
                           (y_err+self._err_offset[1]) if y_phase else 0,
                           -self._control_p *
                           (z_err+self._err_offset[2]) if z_phase else 0,
                           0,
                           1,])
        # action = np.array([0,0,2,0,0])
        action = np.clip(action, -np.ones(action.shape), np.ones(action.shape))
        # print("action is ", action)
        return action

    @property
    def observation_space(self):
        obs = {k: v for k, v in self.env.observation_space.items()}
        obs['controller_state'] = gym.spaces.Box(low=0,
                                                 high=1, shape=(1,), dtype=np.float32)
        return gym.spaces.Dict(obs)
