import gym
from gym_ras.env.embodied.dvrk.rgbd_dvrk import RGBD_CAM
from gym_ras.env.embodied.dvrk.oracle import OracleInput
from gym_ras.tool.config import load_yaml
import numpy as np
import yaml
import time
import time

class GraspAny(gym.Env):
    def __init__(self,
                 arm_names=["PSM1",],
                 rgbd_device="rs435",
                 done_device="",
                 ws_x=[-0.1, 0.1],
                 ws_y=[-0.1, 0.1],
                 ws_z=[-0.24, 0],
                 psm_reset_q=[0, 0, 0.12, 0, 0, 0],
                 psm_open_gripper_deg=40,
                 psm_init_gripper_quat=[7.07106781e-01,  7.07106781e-01, 0, 0],
                 psm_init_pose_low_gripper=[-0.5, -0.5, -0.5, -0.9],
                 psm_init_pose_high_gripper=[0.5, 0.5, 0.5, 0.9],
                 psm_max_step_pos=0.01,
                 psm_max_step_rot=20,
                 cam_image_height=600,
                 cam_image_width=600,
                 cam_depth_remap_center=None,
                 cam_depth_remap_range=None,
                 cam_segment_tool="",
                 cam_segment_model_dir="",
                 dvrk_cal_file='',
                 cam_cal_file='',
                 done_cal_file='',
                 cam_mask_noisy_link=True,
                 reset_random_pose=True,
                 grasp_lift=0.03,
                 ):
        self._arm_names = arm_names
        self._arms = {}
        self._seed = 0
        self._reset_random_pose = reset_random_pose
        self._grasp_lift = grasp_lift
        if done_cal_file == '':
            self._done_tip_z_thres = -1.0
            self._done_jaw_thres = -1.0
        else:
            _args = load_yaml(done_cal_file)
            self._done_tip_z_thres = _args['done_tip_z_thres']
            self._done_jaw_thres = _args['done_jaw_thres']

        for name in arm_names:
            if name in ["PSM1", "PSM2"]:
                psm_args = {
                    "arm_name": name,
                    "ws_x": ws_x,
                    "ws_y": ws_y,
                    "ws_z": ws_z,
                    # "action_mode": 'yaw',
                    "reset_q": psm_reset_q,
                    "open_gripper_deg": psm_open_gripper_deg,
                    "init_gripper_quat": psm_init_gripper_quat,
                    "init_pose_low_gripper": psm_init_pose_low_gripper,
                    "init_pose_high_gripper": psm_init_pose_high_gripper,
                    "max_step_pos": psm_max_step_pos,
                    "max_step_rot": psm_max_step_rot,
                }

                if dvrk_cal_file != '':
                    add_args = load_yaml(dvrk_cal_file)
                    psm_args.update(add_args)
                from gym_ras.env.embodied.dvrk.psm_dvrk import SinglePSM
                self._arms[name] = SinglePSM(
                    **psm_args
                )
            else:
                raise NotImplementedError
        cam_arg = {
            "device": rgbd_device,
            "image_height": cam_image_height,
            "image_width": cam_image_width,
            "depth_remap_center": cam_depth_remap_center,
            "depth_remap_range": cam_depth_remap_range,
            "segment_tool": cam_segment_tool,
            "segment_model_dir": cam_segment_model_dir,
            "mask_noisy_link": cam_mask_noisy_link,
        }
        self._cam_depth_remap_center = cam_depth_remap_center
        self._cam_depth_remap_range = cam_depth_remap_range
        if cam_cal_file != '':
            add_args = load_yaml(cam_cal_file)
            cam_arg.update(add_args)
        self._cam_device = RGBD_CAM(**cam_arg
                                    )
        


        self._done_device_name = done_device
        if done_device == "":
            self._done_device = None
        elif done_device == "keyboard":
            from gym_ras.tool.keyboard import Keyboard
            self._done_device = Keyboard(blocking=True)
        elif done_device == "ds":
            from gym_ras.tool.ds import DS_Controller
            self._done_device = DS_Controller(wait_hz=10, only_press=True)
            
        else:
            raise NotImplementedError




    def render(self):
        return self._cam_device.render()


    def step(self, action, is_out):
        _psm = self._arms[self._arm_names[0]]
        # if not is_out: 
        _psm.step(action)
        time.sleep(1) # render lagging
        obs = _psm.get_obs()
        reward = 0
        info = {}
        done = False

        return obs, reward, done, info

    def _fsm(self, info):
        # if self._done_device is not None:
        #     if self._done_device_name == "keyboard":
        #         ch = self._done_device.get_char()
        #         self._done_device.reset_char_buffer()
        #         is_grasp = ch == "g"

        # else:
        #     if self._done_jaw_thres == -1.0 or self._done_tip_z_thres == -1.0:
        #         return False

        #     _psm = self._arms[self._arm_names[0]]
        #     # f = _psm.jaw_force
        #     # is_grasp = np.abs(f)>np.abs(self._done_jaw_thres)
        #     f = np.rad2deg(_psm.jaw_pos)
        #     is_grasp = np.abs(f) < np.abs(self._done_jaw_thres)
        _psm = self._arms[self._arm_names[0]]
        jaw_pos = np.rad2deg(_psm.jaw_pos)
        is_grasp = jaw_pos < 5

        ws = self.workspace_limit
        _low = ws[:, 0]
        _z_low = _low[2]
        z_current = self.get_prio_obs()["robot_prio"][2]
        is_lift = z_current - _z_low > self._done_tip_z_thres
        # print(z_current, _z_low, z_current - _z_low)
        # print("Grasp:",is_grasp, " is_lift:" , is_lift)
        info['fsm'] = "prog_norm"
        done = is_grasp
        if done:
            _psm = self._arms[self._arm_names[0]]
            _psm.motion_lift(self._grasp_lift, jaw_close=True)
            while True:
                ch = self._done_device.get_char()
                if ch in ["f","j"]:
                    if ch == "f":
                        info['fsm'] = "done_success"
                    else:
                        info['fsm'] = "done_fail"
                    break
            _psm.open_gripper()
            ch = self._done_device.get_char() # wait for manual reset
        
        return info
    
    def psm_reset_pose(self):
        _psm = self._arms[self._arm_names[0]]
        _psm.reset_pose()

    def reset_pose(self):
        _psm = self._arms[self._arm_names[0]]
        _psm.reset_pose()
        _psm.move_gripper_init_pose()

    def reset(self,):
        _psm = self._arms[self._arm_names[0]]
        _psm.reset_pose()
        time.sleep(1)
        self._cam_device._segment.reset()
        _psm.move_gripper_init_pose()
        return _psm.get_obs()

    @property
    def observation_space(self):
        space = self._arms[self._arm_names[0]].obs_space
        obs = {}
        obs['gripper_state'] = gym.spaces.Box(
            space['gripper_state'][0], space['gripper_state'][1], (1,), dtype=np.float32)
        ws = space['tip_pos']
        _low = ws[:, 0]
        _high = ws[:, 1]
        obs['robot_prio'] = gym.spaces.Box(_low, _high, dtype=np.float32)
        return gym.spaces.Dict(obs)

    @property
    def workspace_limit(self):
        return self._arms[self._arm_names[0]].workspace_limit

    def get_prio_obs(self):
        _psm = self._arms[self._arm_names[0]]
        obs = _psm.get_obs()
        return obs

    @property
    def max_step_pos(self,):
        _psm = self._arms[self._arm_names[0]]
        return _psm._max_step_pos

    @property
    def action_space(self,):
        if len(self._arm_names) == 1:
            low_high = self._arms[self._arm_names[0]].act_space
        else:
            raise NotImplementedError
        return gym.spaces.Box(low=low_high[0], high=low_high[1])

    def __del__(self):
        for k, v in self._arms.items():
            del v

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        _seed = seed
        for k, v in self._arms.items():
            _seed -= 1
            v.seed = _seed

    @property
    def reward_dict(self):
        return {
            "done_success": 0,
            "done_fail": 0,
            "prog_norm": 0,
            "prog_abnorm_1": 0,
            "prog_abnorm_2": 0,
            "prog_abnorm_3": 0,
        }

    @property
    def nodepth_guess_map(self,):
        return {
            "psm1": "psm1_except_gripper",
        }

    @property
    def nodepth_guess_uncertainty(self,):
        return {
            "psm1": 0.01,  # 1cm
        }

    @property
    def depth_remap_range(self,):
        c = self._cam_depth_remap_center
        r = self._cam_depth_remap_range / 2
        return [(c-r, c+r)]


if __name__ == "__main__":
    env = NeedlePick()
    obs = env.reset()

    # for i in range(10):
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)

    from gym_ras.env.wrapper import Visualizer
    env = Visualizer(env, update_hz=100)
    _ = env.reset()
    img = env.render()
    img_break = env.cv_show(imgs=img)
    for i in range(20):
        img = env.render()
        # action = env.action_space.sample()
        action = env.get_oracle_action()
        obs, reward, done, info = env.step(action)
        img_break = env.cv_show(imgs=img)
        if img_break:
            break
