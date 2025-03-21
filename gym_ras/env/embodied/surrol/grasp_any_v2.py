import os
import time
import numpy as np

import pybullet as p
from gym_ras.env.embodied.surrol.psm_env import PsmEnv
from gym_ras.env.embodied.surrol.grasp_any_base import GraspAnyBase
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,
    wrap_angle
)
from surrol.const import ASSET_DIR_PATH

from surrol.robots.psm import Psm1


import numpy as np
import pybullet as p
# from surrol.utils.pybullet_utils import get_link_pose, wrap_angle
import os
from gym_ras.tool.common import *
from pathlib import Path


class GraspAnyV2(PsmEnv):
    WORKSPACE_LIMITS1 = (
        (0.50, 0.60),
        (-0.05, 0.05),
        (0.675, 0.675 + 0.07),
    )
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE = 3
    haptic = True
    def __init__(
        self,
        render_mode=None,
        cid=-1,
        # stuff_name="needle",
        fix_goal=True,
        oracle_pos_thres=0.1,
        oracle_rot_thres=1,
        oracle_discrete=True,
        done_z_thres=0.3,
        init_pose_ratio_low_gripper=[-0.5, -0.5, -0.5, -0.9],
        init_pose_ratio_high_gripper=[0.5, 0.5, 0.5, 0.9],
        init_pose_ratio_low_stuff=[-0.5, -0.5, 0.1, -0.99, -0.99, -0.99],
        init_pose_ratio_high_stuff=[0.5, 0.5, 0.5, 0.99, 0.99, 0.99],
        depth_distance=0.25,
        object_scaling_low=0.75,
        object_scaling_high=1.25,
        object_list = ['needle', "box", "bar"],
        needle_prob = 0.5,
        on_plane=False,
        max_grasp_trial=3,
        horizontal_stuff=False,
        **kwargs,
    ):

        _z_level = 0.0025
        self.POSE_TRAY = ((0.55, 0, 0.6751 + _z_level), (0, 0, 0))
        self._init_pose_ratio_low_gripper = init_pose_ratio_low_gripper
        self._init_pose_ratio_high_gripper = init_pose_ratio_high_gripper
        self._init_pose_ratio_low_stuff = init_pose_ratio_low_stuff
        self._init_pose_ratio_high_stuff = init_pose_ratio_high_stuff
        self._object_scaling_low = object_scaling_low
        self._object_scaling_high = object_scaling_high
        self._object_list = object_list

        
        self._on_plane = on_plane
        self._max_grasp_trial = max_grasp_trial
        self._fix_goal = fix_goal
        self._oracle_discrete = oracle_discrete
        self._horizontal_stuff = horizontal_stuff
        self._needle_prob = needle_prob

        super().__init__(render_mode, cid)
        self._view_param = {
            "distance": depth_distance * self.SCALING,
            "yaw": 180,
            "pitch": -20,
            "roll": 0,
        }
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING,
                                  0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        self._proj_param = {"fov": 45, "nearVal": 0.001, "farVal": 1000} # large fov to make sure large focus value, to rectify image without distortion
        self._oracle_pos_thres = oracle_pos_thres
        self._oracle_rot_thres = oracle_rot_thres
        self._done_z_thres = done_z_thres

    def _psm_env_setup(self, stuff_path,   tray_path, scaling, on_plane, goal_plot=False):
        # # camera
        # if self._render_mode == 'human':
        #     reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
        #                  target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))

        # robot
        self.psm1 = Psm1(self.POSE_PSM1[0], p.getQuaternionFromEuler(self.POSE_PSM1[1]),
                         scaling=self.SCALING)
        self.psm1_eul = np.array(p.getEulerFromQuaternion(
            self.psm1.pose_rcm2world(self.psm1.get_current_position(), 'tuple')[1]))  # in the world frame
        if self.ACTION_MODE in ['yaw', 'yaw_tilt']:
            self.psm1_eul = np.array([np.deg2rad(-90), 0., self.psm1_eul[2]])
        elif self.ACTION_MODE == 'pitch':
            self.psm1_eul = np.array([np.deg2rad(0), self.psm1_eul[1], np.deg2rad(-90)])
        else:
            raise NotImplementedError
        self.psm2 = None
        self._contact_constraint = None
        self._contact_approx = False

        # p.loadURDF(os.path.join(ASSET_DIR_PATH, 'table/table.urdf'),
        #            np.array(self.POSE_TABLE[0]) * self.SCALING,
        #            p.getQuaternionFromEuler(self.POSE_TABLE[1]),
        #            globalScaling=self.SCALING)

        # for goal plotting
        if goal_plot:
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                                globalScaling=self.SCALING)
            self.obj_ids['fixed'].append(obj_id)  # 0
        else: 
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                                globalScaling=0.01) #visually remove
            self.obj_ids['fixed'].append(obj_id)  # 0    

        ####### some modification

        self.has_object = True
        self._waypoint_goal = True

        # camera
        if self._render_mode == 'human':
            reset_camera(yaw=89.60, pitch=-56, dist=5.98,
                         target=(-0.13, 0.03, -0.94))

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics(
            (pos, orn), self.psm1.EEF_LINK_INDEX)
        # self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # self._contact_approx = False

        # tray pad
        obj_id = p.loadURDF(
            tray_path,
            np.array(self.POSE_TRAY[0]) * self.SCALING,
            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
            useFixedBase=1,
            globalScaling=self.SCALING,
        )
        self.obj_ids['fixed'].append(obj_id)  # 1

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(
            stuff_path,
            (
                workspace_limits[0].mean()
                + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
                workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                workspace_limits[2][0] + 0.01,
            ),
            p.getQuaternionFromEuler((0, 0, yaw)),
            useFixedBase=False,
            globalScaling=self.SCALING * scaling,
        )
        # print("scaling: ", scaling)
        # p.changeVisualShape(
        #     obj_id, -1, rgbaColor=[0, 0.7, 0, 1], specularColor=(80, 80, 80))  # green
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id = self.obj_ids['rigid'][0]
        self.obj_link1 = 1 if "needle" in stuff_path else -1

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 *
                         np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        return goal.copy()


    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx or self.haptic is True:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def _env_setup(self):
        asset_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "asset" / "urdf"
        )

        if self._on_plane:
            file_dir = {
                "needle": [
                    "needle_40mm_RL.urdf",
                ],
                "box": ["bar.urdf", "box.urdf"],
            }
        else:
            file_dir = {
                "needle": ["needle_40mm_RL.urdf",],
                "bar": ["bar.urdf"],
                "box": ["box.urdf"],
                "sphere": ["sphere.urdf"],
            }

        # needle_dir = file_dir['']
        # file_dir = {k: [asset_path / k / _v for _v in v] for k, v in file_dir.items()}

        # if self._stuff_name == "any":
        #     _dirs = []
        #     for _, v in file_dir.items():
        #         _dirs.extend(v)
        # else:
        #     _dirs = file_dir[self._stuff_name]

        # _stuff_dir = _dirs[self._stuff_urdf_rng.randint(len(_dirs))]
        if self._stuff_urdf_rng.uniform(0,1) > self._needle_prob and "needle" in self._object_list:
            _randir = file_dir["needle"]
            _stuff_file = _randir[self._stuff_urdf_rng.randint(len(_randir))]
            _stuff_dir = asset_path / "needle" / _stuff_file
        else:
            _object_list = [v for v in self._object_list if v!="needle"] 
            flag = _object_list[self._stuff_urdf_rng.randint(len(_object_list))]
            flag_list = file_dir[flag]
            _stuff_file = flag_list[self._stuff_urdf_rng.randint(len(flag_list))]
            _stuff_dir = asset_path / flag / _stuff_file

        
        self._urdf_file_name = _stuff_dir.name
        if self._urdf_file_name == "needle_40mm_RL.urdf":
            _low = 0.5 * self._object_scaling_low
            _high = 0.5 * self._object_scaling_high
        elif self._urdf_file_name == "box.urdf":
            _low = 0.5 * self._object_scaling_low
            _high = 0.5 * self._object_scaling_high
        elif self._urdf_file_name == "bar.urdf":
            _low = 0.9 * self._object_scaling_low
            _high = 0.9 * self._object_scaling_high
        elif self._urdf_file_name == "sphere.urdf":
            _low = 0.3 * self._object_scaling_low
            _high = 0.3 * self._object_scaling_high
        else:
            _low = self._object_scaling_low
            _high = self._object_scaling_high

        scaling = self._stuff_urdf_rng.uniform(_low, _high)

        _tray_dir = asset_path / "tray" / "tray_no_collide.urdf"
        self._psm_env_setup(stuff_path=str(_stuff_dir), tray_path=str(_tray_dir), scaling=scaling, on_plane=self._on_plane)

        # set random gripper init pose
        pos_rel = self._gripper_pose_rng.uniform(
            self._init_pose_ratio_low_gripper, self._init_pose_ratio_high_gripper
        )
        ws = self.workspace_limits1
        new_low = np.array([ws[0][0], ws[1][0], ws[2][0], -180])
        new_high = np.array([ws[0][1], ws[1][1], ws[2][1], 180])
        pose = scale_arr(
            pos_rel, -np.ones(pos_rel.shape), np.ones(pos_rel.shape), new_low, new_high
        )
        M = Quat2M([0.5, 0.5, -0.5, -0.5])
        M1 = Euler2M([0, 0, -22.5], convension="xyz", degrees=True)
        M2 = np.matmul(M1, M)
        quat = M2Quat(M2)
        pos = pose[:3]
        joint_positions = self.psm1.inverse_kinematics(
            (pos, quat), self.psm1.EEF_LINK_INDEX
        )
        self.psm1.reset_joint(joint_positions)

        # set random stuff init pose
        pos_rel = self._stuff_pose_rng.uniform(
            self._init_pose_ratio_low_stuff, self._init_pose_ratio_high_stuff
        )
        _z_level = 0 if self._on_plane else -0.1 
        new_low = np.array([ws[0][0], ws[1][0], ws[2][0] + _z_level, -180, -180, -180])
        new_high = np.array([ws[0][1], ws[1][1], ws[2][1] + _z_level, 180, 180, 180])

        pose = scale_arr(
            pos_rel, -np.ones(pos_rel.shape), np.ones(pos_rel.shape), new_low, new_high
        )
        M = Euler2M([0, 0, 0], convension="xyz", degrees=True)
        if self._on_plane:
            M1 = Euler2M([0, 0, pose[5]], convension="xyz", degrees=True)
        else:
            if not self._horizontal_stuff:
                _m1 = Euler2M([pose[3], 0, 0], convension="xyz", degrees=True)
                _m2 = Euler2M([0, pose[4], 0], convension="xyz", degrees=True)
                _m3 = Euler2M([0, 0, pose[5]], convension="xyz", degrees=True)
            else:
                _m1 = Euler2M([-0, 0, 0], convension="xyz", degrees=True)
                _m2 = Euler2M([0, -0, 0], convension="xyz", degrees=True)
                _m3 = Euler2M([0, 0, pose[5]], convension="xyz", degrees=True)

            M1 = np.matmul(_m2,_m1,)
            M1 = np.matmul(_m3, M1,)
        M2 = np.matmul(M1, M)
        quat = M2Quat(M2)
        pos = pose[:3]
        p.resetBasePositionAndOrientation(self.obj_ids["rigid"][0], pos, quat)

        if not self._on_plane:
            body_pose = p.getBasePositionAndOrientation(self.obj_ids["fixed"][1])
            # body_pose = p.getLinkState(self.obj_ids['fixed'][1], -1)
            if self.obj_link1 == -1:
                obj_pose =  p.getBasePositionAndOrientation(self.obj_id)
            else:
                obj_pose = p.getLinkState(self.obj_id, self.obj_link1)
            world_to_body = p.invertTransform(body_pose[0], body_pose[1])
            obj_to_body = p.multiplyTransforms(world_to_body[0],
                                            world_to_body[1],
                                            obj_pose[0], obj_pose[1])       
            self._init_stuff_constraint = p.createConstraint(
                parentBodyUniqueId=self.obj_ids["fixed"][1],
                parentLinkIndex=-1,
                childBodyUniqueId=self.obj_id,
                childLinkIndex=self.obj_link1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=obj_to_body[0],
                parentFrameOrientation=obj_to_body[1],
                childFramePosition=(0, 0, 0),
                childFrameOrientation=(0, 0, 0),
            )

            p.changeConstraint(self._init_stuff_constraint, maxForce=20)

    def reset(self):
        obs = super().reset()
        self._grasp_trial_cnt = 0
        self._init_stuff_z = self._get_stuff_z()
        self._create_waypoint()
        return obs

    def _create_waypoint(self):
        for _ in range(100):
            p.stepSimulation()  # wait stable
        # if self.obj_link1<0:
        #     pos_obj, orn_obj =  p.getBasePositionAndOrientation(self.obj_id)
        #     pos_obj = list(pos_obj)
        #     orn_obj = list(orn_obj)
        # else:
        #     pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1 if self._on_plane  else 2)
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        if not self._on_plane and self._urdf_file_name == "needle_40mm_RL.urdf":
            pos_obj[0] -= 0.03
            pos_obj[1] += 0.01
            pos_obj[2] += 0.06
        elif self._urdf_file_name == "box.urdf":
            pos_obj[0] -= 0.03
            pos_obj[1] += 0.008
            pos_obj[2] += 0.06
        if not self._on_plane and self._urdf_file_name == "bar.urdf":
            pos_obj[0] -= 0.03
            pos_obj[1] += 0.008
            pos_obj[2] += 0.06
        if not self._on_plane and self._urdf_file_name == "bar2.urdf":
            pos_obj[0] -= 0.03
            pos_obj[1] += 0.008
            pos_obj[2] += 0.06


                #     "needle": [
                #     "needle_40mm_RL.urdf",
                # ],
                # "box": ["bar2.urdf", "bar.urdf", "box.urdf"],
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = (
            orn[2]
            if abs(wrap_angle(orn[2] - orn_eef[2]))
            < abs(wrap_angle(orn[2] + np.pi - orn_eef[2]))
            else wrap_angle(orn[2] + np.pi)
        )  #
        self._WAYPOINTS = [None] * 4
        self._WAYPOINTS[0] = np.array(
            [
                pos_obj[0],
                pos_obj[1],
                pos_obj[2] + 0.01 * self.SCALING if self.tilt_angle==0 else pos_obj[2] + 0.002 * self.SCALING,
                yaw,
                1,
            ]
        )  # approachc
        self._WAYPOINTS[1] = np.array(
            [
                pos_obj[0],
                pos_obj[1],
                pos_obj[2] - 0.002 * self.SCALING if self.tilt_angle==0 else pos_obj[2] - 0.006 * self.SCALING,
                yaw,
                1,
            ]
        )  # approach
        self._WAYPOINTS[2] = self._WAYPOINTS[1].copy()  # grasp
        self._WAYPOINTS[2][4] = -1
        self._WAYPOINTS[3] = self._WAYPOINTS[0].copy()
        self._WAYPOINTS[3][4] = -1
        

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self._jump_sig_prv:
            self._grasp_trial_cnt += 1
        return obs, reward, done, info

    def _is_grasp_obj(self):
        return not (self._contact_constraint == None)

    def _get_stuff_z(self):
        stuff_id = self.obj_ids["rigid"][0]
        pos_obj, _ = get_link_pose(stuff_id, -1)
        return pos_obj[2]

    def _fsm(self,):
        if self._on_plane:
            obs = self._get_robot_state(idx=0)  # forc psm1
            tip = obs[2]
            if (
                self._is_grasp_obj()
                and (tip - self.workspace_limits1[2][0]) > self._done_z_thres
            ):
                return "done_success"
            stuff_z = self._get_stuff_z()
            # prevent stuff is lifted without grasping or grasping unrealisticly
            if (not self._is_grasp_obj()) and (
                stuff_z - self._init_stuff_z
            ) > self.SCALING * 0.015:
                return "done_fail"

            return "prog_norm"
        else:
            if self._init_stuff_constraint is not None and self._contact_constraint is None:
                if self._grasp_trial_cnt >= self._max_grasp_trial:
                    return "done_fail"
                else:
                    return "prog_norm"
            elif self._init_stuff_constraint is None and self._contact_constraint is not None:
                return "done_success"
            else:
                return "done_fail"
    def _sample_goal(self) -> np.ndarray:
        """Samples a new goal and returns it."""
        scale = 0 if self._fix_goal else self.SCALING
        workspace_limits = self.workspace_limits1
        goal = np.array(
            [
                workspace_limits[0].mean() + 0.01 * np.random.randn() * scale,
                workspace_limits[1].mean() + 0.01 * np.random.randn() * scale,
                workspace_limits[2][1] - 0.04 * self.SCALING,
            ]
        )
        return goal.copy()

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        delta_yaw = None
        for i, waypoint in enumerate(self._WAYPOINTS):
            # print("waypoint", i)
            if waypoint is None:
                continue
            if i == 4 and (not self._is_grasp_obj()):
                self._create_waypoint()
                return self.get_oracle_action(obs)
            delta_pos = (waypoint[:3] - obs["observation"][:3]) / 0.01 / self.SCALING
            delta_yaw = wrapAngle((waypoint[3] - obs["observation"][5]), angle_range=np.pi/2)
            if not self._urdf_file_name == "needle_40mm_RL.urdf":
                delta_yaw = 0
            delta_yaw = 0
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array(
                [delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]]
            )
            # print(delta_pos * 0.01 / scale_factor, self._oracle_pos_thres)
            # print("oracle debug")
            # print((waypoint[3] - obs["observation"][5]),delta_yaw, self._oracle_rot_thres)
            if (
                np.linalg.norm(delta_pos) * 0.01 / scale_factor < self._oracle_pos_thres
                and np.abs(delta_yaw) < self._oracle_rot_thres
            ):
                self._WAYPOINTS[i] = None
            break
        if delta_yaw is None: 
            delta_yaw = 0
        if self._oracle_discrete and np.abs(delta_yaw) < self._oracle_rot_thres:
            action[3] = 0
        
        return action

    def seed(self, seed=0):
        super().seed(seed)
        self._init_rng(seed)

    def _init_rng(self, seed):
        stuff_pose_seed = np.uint32(seed)
        gripper_pose_seed = np.uint32(seed - 1)
        print("stuff_pose_seed:", stuff_pose_seed)
        print("gripper_pose_seed:", gripper_pose_seed)
        self._stuff_pose_rng = np.random.RandomState(stuff_pose_seed)
        self._gripper_pose_rng = np.random.RandomState(
            gripper_pose_seed
        )  # use different seed
        self._stuff_urdf_rng = np.random.RandomState(stuff_pose_seed)

    @property
    def keyobj_ids(
        self,
    ):
        return {
            "psm1": self.psm1.body,
            "stuff": self.obj_id,
        }

    @property
    def keyobj_link_ids(
        self,
    ):
        return {
            "psm1": [3, 4,5, 6, 7],
            "psm1_except_gripper": [3, 4, 5],
            "stuff": [-1],
        }

    @property
    def nodepth_link_ids(
        self,
    ):
        return {
            "psm1": [6, 7],
        }

    @property
    def nodepth_guess_map(
        self,
    ):
        return {
            "psm1": "psm1_except_gripper",
        }

    @property
    def nodepth_guess_uncertainty(
        self,
    ):
        return {
            "psm1": 0.01 * self.SCALING,  # 1cm x simluation scaling
        }

    @property
    def background_obj_ids(
        self,
    ):
        return {
            "tray": self.obj_ids["fixed"][1],
        }

    @property
    def background_obj_link_ids(
        self,
    ):
        return {
            "tray": [-1, 4],
        }

    @property
    def random_vis_key(self):
        return ["tray", "psm1", "stuff"]

    @property
    def random_color_range(
        self,
    ):
        return "hsv", {
            "default": [[0.4, 0.4, 0.4, 1], [0.4, 1, 0.4, 1]],
            "tray": [[0, 0.0, 0.7, 1], [1, 0.2, 1, 1]],
            "psm1": [[0.7, 0.0, 0.7, 1], [1, 0.2, 1, 1]],
            "stuff": [[0.0, 0, 0.5, 1], [1, 0.4, 1, 1]],
        }
