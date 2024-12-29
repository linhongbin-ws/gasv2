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
        debug=False,
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
        self._phase_pid = True
        self._prv_sigs = {}
        self._debug = debug

    def reset(self):
        obs = self.env.reset()
        pid_obs = self._get_pid_observation(obs['occup_mat'])
        pid_phase, x_phase, y_phase, z_phase = self._check_pid_phase(pid_obs)
        obs["controller_state"] = 1 if self._phase_pid else 0
        self._prv_sigs['x_phase'] = x_phase
        self._prv_sigs['y_phase'] = y_phase
        self._prv_sigs['z_phase'] = z_phase
        self._prv_sigs['pid_obs'] = pid_obs
        self._phase_pid = pid_phase and (not self._skip)
        return obs

    def step(self, action):
        if self._phase_pid:
            action = self._get_pid_action(self._prv_sigs['pid_obs'], self._prv_sigs['x_phase'], self._prv_sigs['y_phase'], self._prv_sigs['z_phase'])
        obs, reward, done, info = self.env.step(action)
        pid_obs = self._get_pid_observation(obs['occup_mat'])
        pid_phase, x_phase, y_phase, z_phase = self._check_pid_phase(pid_obs)
        obs["controller_state"] = 1 if self._phase_pid else 0
        self._prv_sigs['x_phase'] = x_phase
        self._prv_sigs['y_phase'] = y_phase
        self._prv_sigs['z_phase'] = z_phase
        self._prv_sigs['pid_obs'] = pid_obs
        self._phase_pid = pid_phase and (not self._skip)
        info = self._fsm(info, pid_obs)
        return obs, reward, done, info

    def _fsm(self, info, pid_obs, ):
        if pid_obs['err'][2] < self._fsm_z_err_min:
            if info['fsm'].find("done") < 0:
                info['fsm'] =  self._fsm_z_err_state
            if self._debug: print(f"exceed z err, err: {pid_obs['err'][2]}, thres: {self._fsm_z_err_min}")
        return info

    def _get_pid_observation(self, occup_mat):
        if self._obs_type == 'occup':
            occ = occup_mat
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
