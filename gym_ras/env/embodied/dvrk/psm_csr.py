from gym_ras.tool.common import TxT, invT, getT, T2Quat, scale_arr, M2Euler, wrapAngle, Euler2M, printT, Quat2M, M2Quat
from gym_ras.tool.kdl_tool import Frame2T, Quaternion2Frame, gen_interpolate_frames, T2Frame
import numpy as np
import gym
from csrk.arm_proxy import ArmProxy
from csrk.node import Node
import PyCSR
from gym_ras.tool.csr_tool import T2CsrFrame, CsrFrame2T
import time
from gym_ras.tool.psm_smoother import TSmooth

class SinglePSM():
    ACTION_SIZE = 5

    def __init__(self,
                 action_mode='yaw',
                 arm_name='PSM1',
                 max_step_pos=0.02,
                 max_step_rot=40,
                 open_gripper_deg=57,
                 init_gripper_quat=[7.07106781e-01,  7.07106781e-01, 0, 0],
                 init_pose_low_gripper=[-0.5, -0.5, -0.5, -0.9],
                 init_pose_high_gripper=[0.5, 0.5, 0.5, 0.9],
                 ws_x=[-0.1, 0.1],
                 ws_y=[-0.1, 0.1],
                 ws_z=[-0.24, 0],
                 world2base_yaw=0,
                 reset_q=[0, 0, 0.12, 0, 0, 0],
                 robot="csr",
                 ):
        assert arm_name in ["PSM1", "PSM2", "PSM3",]
        node_ = Node("NDDS_QOS_PROFILES.CSROS.xml") # NOTE: path Where you put the ndds xml file
        # ecm = ArmProxy(node_, "psa2")
        # psa1 = ArmProxy(node_, "psa1")
        self._psm = ArmProxy(node_, "psa3")

        while(not self._psm.is_connected):
            self._psm.measured_cp()
            # To check if the arm is connected
            self._psm.read_rtrk_arm_state()
            print("connection: ",self._psm.is_connected)

        from gym_ras.env.embodied.dvrk.psm_controller import PSM_Controller
        self._psm = PSM_Controller()
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
        self._smoother = TSmooth(alpha=1)

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

    def moveT(self, T, jaw_deg, interp_num=-1,duration=1):
        T_local = TxT([invT(self._world2base), T])
        self.moveT_local(T_local, jaw_deg, interp_num,duration=duration)

    def moveT_local(self, T, jaw_deg, interp_num=-1,duration=1, block=True):
        T_origin = self.tip_pose_local
        frame1 = T2Frame(T_origin)
        frame2 = T2Frame(T)
        if interp_num>0:
            frames = gen_interpolate_frames(frame1, frame2,num=interp_num)
            for i, f in enumerate(frames):
                self._psm.move_cp(T2CsrFrame(Frame2T(f)), acc=1, duration=0.3, jaw=np.deg2rad(jaw_deg))
                time.sleep(0.3)
            # print(T2CsrFrame(Frame2T(f)))
        else:
            self._psm.move_cp(T2CsrFrame(T), acc=1, duration=duration, jaw=np.deg2rad(jaw_deg))
            # print(T2CsrFrame(T))
            if block:
                time.sleep(duration)

    def move_gripper_init_pose(self):
        pos_rel = self._gripper_pose_rng.uniform(
            self._init_pose_low_gripper, self._init_pose_high_gripper)
        ws = self.workspace_limit
        new_low = np.array([ws[0][0], ws[1][0], ws[2][0], -180])
        new_high = np.array([ws[0][1], ws[1][1], ws[2][1], 180])
        # print("workspace:", ws)

        pose = scale_arr(pos_rel, -np.ones(pos_rel.shape),
                         np.ones(pos_rel.shape), new_low, new_high)
        # print("pose,", pose)
        M = Quat2M(self._init_gripper_quat)

        M1 = Euler2M([0, 0, pose[3]], convension="xyz", degrees=True)
        M2 = np.matmul(M1, M)
        quat = M2Quat(M2)
        pos = pose[:3]
        T = getT(pos_list=pos, rot_list=quat,)
        action_rcm = TxT([invT(self._world2base), T])
        # pos, quat = T2Quat(action_rcm)
        # frame = Quaternion2Frame(*pos, *quat)

        self.moveT_local(action_rcm, jaw_deg=self._open_gripper_deg, interp_num=-1, duration=2)
        # self._psm.move_cp(frame).wait()

    def reset_pose(self):
        q = np.deg2rad(np.array(self._reset_q))
        q[2] = self._reset_q[2]
        q_j = q.tolist() + [np.deg2rad(self._open_gripper_deg)]
        self._psm.move_jp(q_j,max_vel=3, acc=0.2)
        print("reset pose ...", self._psm.measured_jp())
        time.sleep(2)

    def step(self, action):
        self._set_action(action)

    def get_obs(self):
        tip_pos, tip_quat = T2Quat(self.tip_pose)
        gripper_state = np.rad2deg(self._psm.measured_jp()[6])

        obs = {}
        obs["robot_prio"] = tip_pos
        obs["gripper_state"] = gripper_state
        return obs
    # def close_gripper(self,):
    #     self._psm.jaw.close().wait()
    # def open_gripper(self,):
    #     self._psm.jaw.move_jp(np.deg2rad(self._open_gripper_deg)).wait()

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
    def act_space(self):
        return [-np.ones(self.ACTION_SIZE), np.ones(self.ACTION_SIZE)]

    @property
    def tip_pose_quat(self):
        return T2Quat(self.tip_pose)

    @property
    def tip_pose(self):
        return TxT([self._world2base, self.tip_pose_local])

    @property
    def tip_pose_local(self):
        T = CsrFrame2T(self._psm.measured_cp())
        # T = Frame2T(self._psm.setpoint_cp())
        # printT(T, prefix_string="tip_pose_local")
        return T

    def _set_action(self, action: np.ndarray):
        """
        delta_position (3), delta_theta (1) and open/close the gripper (1)
        in the world frame
        """
        assert len(
            action) == self.ACTION_SIZE, "The action should have the save dim with the ACTION_SIZE"
        action = action.copy()  # ensure that we don't change the action outside of this scope

        # position, limit maximum change in position
        action[:3] *= self._max_step_pos
        pose_world = self.tip_pose
        pose_world[:3, 3] = np.clip(
            pose_world[:3, 3] + action[:3], self.workspace_limit[:, 0], self.workspace_limit[:, 1])
        rot = M2Euler(pose_world[:3, :3], convension="xyz", degrees=False)
        if self._action_mode == 'yaw':
            # yaw, limit maximum change in rotation
            action[3] *= np.deg2rad(self._max_step_rot)
            rot[2] = wrapAngle(rot[2] + action[3], degrees=False,
                               angle_range=180)  # only change yaw
            # print(np.rad2deg(rot[2]))
        elif self._action_mode == 'pitch':
            # action[3] *= np.deg2rad(self._max_step_rot)  # pitch, limit maximum change in rotation
            # pitch = np.clip(wrapAngle(rot[1] + action[3]), np.deg2rad(-90), np.deg2rad(90))
            # rot = (self.psm1_eul[0], pitch, self.psm1_eul[2])  # only change pitch
            raise NotImplementedError
        else:
            raise NotImplementedError
        pose_world[:3, :3] = Euler2M(rot, convension="xyz", degrees=False)
        action_rcm = TxT([invT(self._world2base), pose_world])
        # time1 = time.time()
        # printT(action_rcm, "moveT")
        pos, quat = T2Quat(action_rcm)
        pos = np.array(pos)
        pos = pos.tolist()
        goal = Quaternion2Frame(*pos, *quat)

        # self._psm.move_cp(goal).wait()
       

        # jaw
        if action[4] < 0:
            self.moveT_local(self._smoother.smooth(Frame2T(goal)), np.rad2deg(-0.6),  interp_num=-1, duration=0.8, block=False)
            # print("kkkkkkkkkk", np.rad2deg(-0.6))
        else:
            self.moveT_local(self._smoother.smooth(Frame2T(goal)), self._open_gripper_deg, interp_num=-1 , duration=0.8, block=False)
            # open jaw angle; can tune
            # self._psm.close_jaw(1)
        time.sleep(0.9) # too fast, need to waitqqwqqqq

    def __del__(self):
        del self._psm


if __name__ == "__main__":
    from gym_ras.tool.config import load_yaml
    dvrk_cal_file = "./data/dvrk_cal/dvrk_cal.yaml"
    add_args = load_yaml(dvrk_cal_file)
    


    p = SinglePSM(arm_name="PSM1", max_step_pos=0.01, max_step_rot=20, **add_args)
    print("reset q: ", p._reset_q)
    print("ws :", p._ws_x, p._ws_y, p._ws_z)

    # p.reset_pose()
    # # print(p.tip_pose_local)
    # p.move_gripper_init_pose()

    # obs = p.get_obs()
    # print("get obs:", obs)
    for i in range(2):
        p.reset_pose()
        p.move_gripper_init_pose()



    # print(p.tip_pose_quat)
    # action = np.zeros(5, dtype=np.float32)
    # action[4] += 1
    # p.step(action)
    # print("open Jaw")
    # action = np.zeros(5, dtype=np.float32)
    # action[4] -= 1
    # p.step(action)
    # print("close Jaw")


    # print("move x +1")
    # action = np.zeros(5, dtype=np.float32)
    # action[0] += 1
    # p.step(action)

    # print("move x -1")
    # action = np.zeros(5, dtype=np.float32)
    # action[0] -= 1
    # p.step(action)


    # print("move y +1")
    # action = np.zeros(5, dtype=np.float32)
    # action[1] += 1
    # p.step(action)
    # print("move y -1")
    # action = np.zeros(5, dtype=np.float32)
    # action[1] -= 1
    # p.step(action)


    # print("move z +1")
    # action = np.zeros(5, dtype=np.float32)
    # action[2] += 1
    # p.step(action)
    # print("move z -1")
    # action = np.zeros(5, dtype=np.float32)
    # action[2] -= 1
    # p.step(action)

    # print("rot z +1")
    # action = np.zeros(5, dtype=np.float32)
    # action[3] += 1
    # p.step(action)
    # print("rot z -1")
    # action = np.zeros(5, dtype=np.float32)
    # action[3] -= 1
    # p.step(action)


