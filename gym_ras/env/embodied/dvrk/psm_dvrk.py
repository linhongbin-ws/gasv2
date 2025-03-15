
from dvrk import psm
from gym_ras.tool.common import TxT, invT, getT, T2Quat, scale_arr, M2Euler, wrapAngleRange, Euler2M, printT, Quat2M, M2Quat, T2Euler
from gym_ras.tool.kdl_tool import Frame2T, Quaternion2Frame, gen_interpolate_frames, T2Frame
import numpy as np


class SinglePSM():
    ACTION_SIZE = 5
    def __init__(self,
                 action_mode='yaw_tilt',
                 arm_name='PSM1',
                 max_step_pos=0.02,
                 max_step_rot=20,
                 open_gripper_deg=40,
                 init_gripper_quat=[7.07106781e-01,  7.07106781e-01, 0, 0],
                 init_pose_low_gripper=[-0.5, -0.5, -0.5, -0.9],
                 init_pose_high_gripper=[0.5, 0.5, 0.5, 0.9],
                 ws_x=[-0.1, 0.1],
                 ws_y=[-0.1, 0.1],
                 ws_z=[-0.24, 0],
                 world2base_yaw=0,
                 reset_q=[0, 0, 0.12, 0, 0, 0],
                 ):
        assert arm_name in ["PSM1", "PSM2", "PSM3",]
        if arm_name == "PSM1":
            self.TITLT_ANGLE = 45
            self.YAW_LOW = -180
            self.YAW_HIGH = -0
        elif arm_name == "PSM2":
            self.TITLT_ANGLE = -45
            self.YAW_LOW = -270
            self.YAW_HIGH = -90

        self._psm = psm(arm_name)
        self._action_mode = action_mode
        self._world2base = getT([0, 0, 0], [0, 0, world2base_yaw],
                                rot_type="euler", euler_convension="xyz", euler_Degrees=True)
        self._max_step_pos = max_step_pos
        self._max_step_rot = max_step_rot
        self._ws_x = ws_x
        self._ws_y = ws_y
        self._ws_z = ws_z
        self._reset_q = reset_q
        self._open_gripper_deg = open_gripper_deg
        self._init_gripper_quat = init_gripper_quat
        self._init_pose_low_gripper = init_pose_low_gripper
        self._init_pose_high_gripper = init_pose_high_gripper
        self.seed = 0
        self._tip_pose_local = None

        self._gripper_pos = None
        self._gripper_yaw = None
        if self._action_mode == 'yaw_tilt':
            tilt_mat1 = getT([0,0,0], [0,self.TITLT_ANGLE, 0], rot_type="euler", euler_Degrees=True)
            tilt_mat2 = getT([0,0,0], [0, 0, 0], rot_type="euler", euler_Degrees=True)
            self._tilt_mat = np.matmul(tilt_mat1, tilt_mat2)

        
    def move_gripper_init_pose(self):

        # generate init pose
        pos_rel = self._gripper_pose_rng.uniform(
            self._init_pose_low_gripper, self._init_pose_high_gripper)
        ws = self.workspace_limit
        new_low = np.array([ws[0][0], ws[1][0], ws[2][0], self.YAW_LOW])
        new_high = np.array([ws[0][1], ws[1][1], ws[2][1], self.YAW_HIGH])
        pose = scale_arr(pos_rel, -np.ones(pos_rel.shape),
                         np.ones(pos_rel.shape), new_low, new_high)
        self._gripper_pos = pose[:3]
        self._gripper_yaw = self.YAW_HIGH - 22.5 - 90
        T = self._get_T_from_pos_yaw()

        # move to catesian pose
        self.moveT(T, interp_num=-1)

    def _get_T_from_pos_yaw(self):
        M = Quat2M(self._init_gripper_quat)
        M1 = Euler2M([0, 0, self._gripper_yaw], convension="xyz", degrees=True)
        M2 = np.matmul(M1, M)
        if self._action_mode == 'yaw_tilt':
            _T = np.eye(4)
            _T[:3,:3] = M2
            M2 = TxT([_T, self._tilt_mat])[:3,:3]
        T = np.eye(4)
        T[:3,:3] = M2
        T[:3, 3] = self._gripper_pos
        return T

    def reset_pose(self):
        q = np.deg2rad(np.array(self._reset_q))
        q[2] = self._reset_q[2]
        self._psm.jaw.move_jp(np.deg2rad(self._open_gripper_deg)).wait()
        self._psm.move_jp(q).wait()

    def _set_action(self, action):
        assert len(action) == self.ACTION_SIZE, "The action should have the save dim with the ACTION_SIZE"
        action = action.copy()

        if self._action_mode in ['yaw', 'yaw_tilt']:
            new_gripper_pos =  self._gripper_pos + action[:3]*self._max_step_pos
            new_gripper_pos = np.clip(new_gripper_pos, self.workspace_limit[:, 0], self.workspace_limit[:, 1])
            new_gripper_yaw = self._gripper_yaw + action[3]*self._max_step_rot
            new_gripper_yaw = wrapAngleRange(new_gripper_yaw, self.YAW_LOW,  self.YAW_HIGH)
            self._gripper_pos = new_gripper_pos
            self._gripper_yaw = new_gripper_yaw
            T = self._get_T_from_pos_yaw()
        else:
            raise NotImplementedError

        # actuate
        self.moveT(T, interp_num=-1, block=False)
        if action[4] < 0:
            self._psm.jaw.close().wait()
        else:
            self._psm.jaw.move_jp(np.deg2rad(self._open_gripper_deg)).wait()

    def moveT(self, T, interp_num=-1, block=True):
        T_local = TxT([invT(self._world2base), T])
        self.moveT_local(T_local, interp_num, block=block)

    def moveT_local(self, T, interp_num=-1, block=True):
        frame2 = T2Frame(T)
        if interp_num>0:
            frame1 = T2Frame(self.tip_pose_local)
            frames = gen_interpolate_frames(frame1, frame2,num=interp_num)
            for i, f in enumerate(frames):
                self._psm.move_cp(f).wait()
        else:
            self._psm.move_cp(frame2).wait() if block else self._psm.move_cp(frame2)
        self.tip_pose_local = T

    def step(self, action):
        self._set_action(action)

    def get_obs(self):
        tip_pos, tip_quat = T2Quat(self.tip_pose)
        gripper_state = np.rad2deg(self._psm.jaw.measured_jp())

        obs = {}
        obs["robot_prio"] = tip_pos
        obs["gripper_state"] = gripper_state
        return obs
    def close_gripper(self,):
        self._psm.jaw.close().wait()
    def open_gripper(self,):
        self._psm.jaw.move_jp(np.deg2rad(self._open_gripper_deg)).wait()

    def motion_lift(self, lift_height, jaw_close=True):
        self._gripper_pos[2] = self._gripper_pos[2] + lift_height
        T = self._get_T_from_pos_yaw()
        self.moveT(T, interp_num=-1, block=False)
        if jaw_close:
            self._psm.jaw.close().wait()
        else:
            self._psm.jaw.move_jp(np.deg2rad(self._open_gripper_deg)).wait()


    @property
    def workspace_limit(self):
        return np.array([self._ws_x, self._ws_y, self._ws_z])

    @property
    def obs_space(self):
        space = {}
        space["tip_pos"] = self.workspace_limit
        space["gripper_state"] = [0, self._open_gripper_deg]
        return space

    @property
    def jaw_force(self):
        f = self._psm.jaw.measured_jf()
        return f

    @property
    def jaw_pos(self):
        f = self._psm.jaw.measured_jp()
        # print(f)
        return f

    @property
    def act_space(self):
        return [-np.ones(self.ACTION_SIZE), np.ones(self.ACTION_SIZE)]


    @property
    def tip_pose(self):
        return TxT([self._world2base, self.tip_pose_local])

    @property
    def tip_pose_local(self):
        if self._tip_pose_local is None:
            self._tip_pose_local = Frame2T(self._psm.setpoint_cp())
        
        return self._tip_pose_local
    
    @tip_pose_local.setter
    def tip_pose_local(self, T):
        self._tip_pose_local = T
    


    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.init_rng(seed)

    def init_rng(self, seed):
        gripper_pose_seed = np.uint32(seed)
        self._gripper_pose_rng = np.random.RandomState(gripper_pose_seed)


    def __del__(self):
        del self._psm


if __name__ == "__main__":
    p = SinglePSM(arm_name="PSM1")
    obs = p.get_obs()
    print("get obs:", obs)
    for i in range(5):
        p.reset_pose()
        p.move_gripper_init_pose()
    print(p.tip_pose_quat)
    action = np.zeros(5, dtype=np.float32)
    action[4] += 1
    p.step(action)
    print("open Jaw")
    action = np.zeros(5, dtype=np.float32)
    action[4] -= 1
    print("close Jaw")
    p.step(action)
    print("move x +1")
    action = np.zeros(5, dtype=np.float32)
    action[0] += 1
    p.step(action)
    print("move x -1")
    action = np.zeros(5, dtype=np.float32)
    action[0] -= 1
    p.step(action)
    print("move y +1")
    action = np.zeros(5, dtype=np.float32)
    action[1] += 1
    p.step(action)
    print("move y -1")
    action = np.zeros(5, dtype=np.float32)
    action[1] -= 1
    p.step(action)
    print("move z +1")
    action = np.zeros(5, dtype=np.float32)
    action[2] += 1
    p.step(action)
    print("move z -1")
    action = np.zeros(5, dtype=np.float32)
    action[2] -= 1
    p.step(action)

    for i in range(100):
        print("move cnt:", i)
        action = np.zeros(5, dtype=np.float32)
        action[0] -= 1
        p.step(action)
