import torch
import torch.nn as nn
import numpy as np
import os
import cv2
# edit for csr
from csrk.arm_proxy import ArmProxy
from csrk.node import Node
import PyCSR
# end edit for csr
# import dvrk
# import PyKDL

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import yaml
import math
from scipy.spatial.transform import Rotation as R
from easydict import EasyDict as edict
import sys
sys.path.append('IGEV/core')
sys.path.append('IGEV')
from igev_stereo import IGEVStereo
from IGEV.core.utils.utils import InputPadder
# from rl.agents.ddpg import DDPG
# import rl.components as components

import argparse
# from FastSAM.fastsam import FastSAM, FastSAMPrompt
import ast
import torch
from PIL import Image
# from FastSAM.utils.tools import convert_box_xywh_to_xyxy

from torchvision.transforms import Compose
import torch.nn.functional as F
import queue, threading

# # from vmodel import vismodel
# from config import opts

from rectify import my_rectify
import time
# edit for csr
node_ = Node("/home/student/csr_test/NDDS_QOS_PROFILES.CSROS.xml") # NOTE: path Where you put the ndds xml file
# end edit for csr

from copy import deepcopy

from gym_ras.tool.common import scale_arr
from depth_remap import depth_remap

def SetPoints(windowname, img):
    
    points = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(temp_img, (x, y), 10, (102, 217, 239), -1)
            points.append([x, y])
            cv2.imshow(windowname, temp_img)

    temp_img = img.copy()
    cv2.namedWindow(windowname)
    cv2.imshow(windowname, temp_img)
    cv2.setMouseCallback(windowname, onMouse)
    key = cv2.waitKey(0)
    if key == 13:  # Enter
        print('select point: ', points)
        del temp_img
        cv2.destroyAllWindows()
        return points
    elif key == 27:  # ESC
        print('quit!')
        del temp_img
        cv2.destroyAllWindows()
        return
    else:
        
        print('retry')
        return SetPoints(windowname, img)

def crop_img(img):
    crop_img = img[:,100: ]
    crop_img = crop_img[:,: -100]
    #print(crop_img.shape)
    crop_img=cv2.resize(crop_img ,(256,256))
    return crop_img

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    video_name='test_record/{}.mp4'.format(name.split('/')[-1])
    # self.output_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (800, 600))
    self.is_alive = True
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()
    #t.join()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while self.is_alive:
      ret, frame = self.cap.read()
      if not ret:
        break
    #   self.output_video.write(frame)
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
  
  def release(self):
      self.is_alive = False
      self.cap.release()
    #   self.output_video.release()


def transf_DH_modified(alpha, a, theta, d):
    trnsf = np.array([[math.cos(theta), -math.sin(theta), 0., a],
                    [math.sin(theta) * math.cos(alpha), math.cos(theta) * math.cos(alpha), -math.sin(alpha), -d * math.sin(alpha)],
                    [math.sin(theta) * math.sin(alpha), math.cos(theta) * math.sin(alpha), math.cos(alpha), d * math.cos(alpha)],
                    [0., 0., 0., 1.]])
    return trnsf

# PSM1


basePSM_T_cam =np.array([[ 0.70426313,  0.67946631, -0.20576438, -0.07995621],
       [ 0.70239444, -0.62472898,  0.34110959, -0.08750956],
       [ 0.1032255 , -0.38475866, -0.91722694, -0.07636903],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

cam_T_basePSM =np.array([[ 0.70426313,  0.70239444,  0.1032255 ,  0.12565967],
       [ 0.67946631, -0.62472898, -0.38475866, -0.02972585],
       [-0.20576438,  0.34110959, -0.91722694, -0.05664952],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])


class VisPlayer(nn.Module):
    def __init__(self):
        super().__init__()
        # depth estimation
        self.device='cuda:0'
        # self._load_depth_model()
        # self._load_policy_model()
        self._init_rcm()
        self.img_size=(320,240)
        self.scaling=1. # for peg transfer
        # edit for csr camera
        self.calibration_data = {
            'baseline': 0.004671,
            'focal_length_left': 788.96950318,
            'focal_length_right': 788.96950318
        }
        # edit for csr camera end
        self.threshold=0.009
        self._segmentor = None
        # self.init_run()

        self.depth_center = 0.14179377
        self.depth_range = 0.1

        self._depth_remap = True



    def _init_rcm(self):
        # TODO check matrix
        self.tool_T_tip=np.array([[ 0. ,-1. , 0. , 0.],
                         [ 0. , 0. , 1. , 0.],
                         [-1. , 0. , 0. , 0.],
                         [ 0. , 0. , 0. , 1.]])

        eul=np.array([np.deg2rad(-90), 0., 0.])
        eul= self.get_matrix_from_euler(eul)
        self.rcm_init_eul=np.array([-2.94573084 , 0.15808114 , 1.1354972])
        # object pos [-0.123593,   0.0267398,   -0.141579]
        # target pos [-0.0577594,   0.0043639,   -0.133283]
        self.rcm_init_pos=np.array([ -0.0617016, -0.00715477,  -0.0661465])

    def _load_depth_model(self, checkpoint_path='pretrained_models/sceneflow.pth'):
        args=edict()
        args.restore_ckpt=checkpoint_path
        args.save_numpy=False
        args.mixed_precision=False
        args.valid_iters=32
        args.hidden_dims=[128]*3
        args.corr_implementation="reg"
        args.shared_backbone=False
        args.corr_levels=2
        args.corr_radius=4
        args.n_downsample=2
        args.slow_fast_gru=False
        args.n_gru_layers=3
        args.max_disp=192

        self.depth_model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
        # self.depth_model=IGEVStereo(args)
        self.depth_model.load_state_dict(torch.load(args.restore_ckpt))

        self.depth_model = self.depth_model.module
        self.depth_model.to("cuda")
        self.depth_model.eval()

    def _load_dam(self):
        encoder = 'vitl' # can also be 'vitb' or 'vitl'
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()
        self.img_transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',    
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            ])

    def _get_depth_with_dam(self, img):
        '''
        input: rgb image 1xHxW
        '''
        img=img/255.0
        h, w = img.shape[:2]

        img=self.img_transform({'image': img})['image']
        img=torch.from_numpy(img).unsqueeze(0)
        with torch.no_grad():
            depth = self.depth_anything(img)

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) # 0-1
        # print(depth.mean())

        depth = depth.cpu().numpy()

        return depth

    def _load_policy_model(self, filepath='./pretrained_models/state_dict.pt'):
        with open('rl/configs/agent/ddpg.yaml',"r") as f:
            agent_params=yaml.load(f.read(),Loader=yaml.FullLoader)
        agent_params=edict(agent_params)
        env_params = edict(
            obs=19,
            achieved_goal=3,
            goal=3,
            act=4,
            max_timesteps=10,
            max_action=1,
            act_rand_sampler=None,
        )

        self.agent=DDPG(env_params=env_params,agent_cfg=agent_params)
        checkpt_path=filepath
        checkpt = torch.load(checkpt_path, map_location='cpu')
        self.agent.load_state_dict(checkpt, strict=False)
        # self.agent.g_norm = checkpt['g_norm']
        # self.agent.o_norm = checkpt['o_norm']
        # self.agent.update_norm_test()
        # print('self.agent.g_norm.mean: ',self.agent.g_norm.mean)
        self.agent.g_norm.std=self.agent.g_norm_v.numpy()
        self.agent.g_norm.mean=self.agent.g_norm_mean.numpy()
        self.agent.o_norm.std=self.agent.o_norm_v.numpy()
        self.agent.o_norm.mean=self.agent.o_norm_mean.numpy()
        # print('self.agent.g_norm.mean: ',self.agent.g_norm.mean)
        # exit()

        '''
        
        self.agent.depth_norm.std=self.agent.d_norm_v.numpy()
        self.agent.depth_norm.mean=self.agent.d_norm_mean.numpy()
        s
        #print(self.agent.g_norm_v)
        '''
        self.agent.cuda()
        self.agent.eval()

        opts.device='cuda:0'
        self.v_model=vismodel(opts)
        ckpt=torch.load(opts.ckpt_dir, map_location=opts.device)
        self.v_model.load_state_dict(ckpt['state_dict'],strict=False)
        self.v_model.to(opts.device)
        self.v_model.eval()

    def convert_disparity_to_depth(self, disparity, baseline, focal_length):
        depth = baseline * focal_length/ disparity
        return depth

    def _get_depth(self, limg, rimg):
        # input image should be RGB(Image.open().convert('RGB')); numpy.array
        '''
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)
        '''
        limg=torch.from_numpy(limg).permute(2, 0, 1).float().to(self.device).unsqueeze(0)
        rimg=torch.from_numpy(rimg).permute(2, 0, 1).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            # print(limg.ndim)
            padder = InputPadder(limg.shape, divis_by=32)
            image1, image2 = padder.pad(limg, rimg)
            disp = self.depth_model(image1, image2, iters=32, test_mode=True)
            disp = disp.cpu().numpy()

            disp = padder.unpad(disp).squeeze()
            depth_map = self.convert_disparity_to_depth(disp, self.calibration_data['baseline'], self.calibration_data['focal_length_left'])
        # return disp
        return depth_map

    def _load_fastsam(self, model_path="./FastSAM/weights/FastSAM-x.pt"):

        self.seg_model=FastSAM(model_path)

    def _seg_with_fastsam(self, input, object_point):
        point_prompt=str([object_point,[200,200]])
        point_prompt = ast.literal_eval(point_prompt)
        point_label = ast.literal_eval("[1,0]")
        everything_results = self.seg_model(
            input,
            device=self.device,
            retina_masks=True,
            imgsz=608,
            conf=0.25,
            iou=0.7    
            )

        prompt_process = FastSAMPrompt(input, everything_results, device=self.device)
        ann = prompt_process.point_prompt(
            points=point_prompt, pointlabel=point_label
        )

        return ann[0]

    def _get_visual_state(self, seg, depth, robot_pos, robot_rot, jaw, goal):
        seg_d=np.concatenate([seg.reshape(1, self.img_size[0], self.img_size[1]), \
                              depth.reshape(1, self.img_size[0], self.img_size[1])],axis=0)

        inputs=torch.tensor(seg_d).unsqueeze(0).float().to(self.device)
        robot_pos=torch.tensor(robot_pos).to(self.device)
        robot_rot=torch.tensor(robot_rot).to(self.device)
        jaw=torch.tensor(jaw).to(self.device)
        goal=torch.tensor(goal).to(self.device)

        with torch.no_grad():
            # print(inputs.shape)
            v_output=self.agent.v_processor(inputs).squeeze()

            waypoint_pos_rot=v_output[3:]

        return waypoint_pos_rot[:3].cpu().data.numpy().copy(), waypoint_pos_rot[3:].cpu().data.numpy().copy()

    def _get_action(self, seg, depth, robot_pos, robot_rot, jaw, goal):
        # the pos should be in ecm space
        '''
        input: seg (h,w); depth(h,w); robot_pos 3; robot_rot 3; jaw 1; goal 3
        '''
        # depth=self.agent.depth_norm.normalize(depth.reshape(self.img_size*self.img_size),device=self.device).reshape(self.img_size,self.img_size)
        # plt.imsave('test_record/pred_depth_norm_{}.png'.format(count),depth)

        # image = self.img_transform({'image': rgb})['image']

        seg=torch.from_numpy(seg).to("cuda:0").float()
        depth=torch.from_numpy(depth).to("cuda:0").float()

        # seg_d=np.concatenate([seg.reshape(1, self.img_size[0], self.img_size[1]), \
        #                      depth.reshape(1, self.img_size[0], self.img_size[1])],axis=0)

        # inputs=torch.tensor(seg_d).unsqueeze(0).float().to(self.device)
        # image=torch.from_numpy(image).to(self.device).float()
        # seg=torch.from_numpy(seg).to(self.device).float()
        # with torch.no_grad():
        #    v_output=self.v_model.get_obs(seg.unsqueeze(0), image.unsqueeze(0))[0]#.cpu().data().numpy()

        robot_pos=torch.tensor(robot_pos).to(self.device)
        robot_rot=torch.tensor(robot_rot).to(self.device)
        # robot_rot=torch.tensor([0.9744, -0.009914,-0.000373]).to(self.device)
        # jaw=torch.tensor([0.6981]).to(self.device)
        jaw=torch.tensor(jaw).to(self.device)
        # print(jaw.shape)
        goal=torch.tensor(goal).to(self.device)

        with torch.no_grad():
            # print(inputs.shape)
            # v_output=self.agent.v_processor(inputs).squeeze()
            v_output=self.v_model.get_obs(seg.unsqueeze(0), depth.unsqueeze(0))[0]
            # goal=v_output[-3:]
            # goal=torch.tensor(goal).to(self.device)
            # v_output=v_output[:-3]
            # print(v_output)
            # save_v=v_output.cpu().data.numpy()
            # np.save('test_record/v_output.npy',save_v)
            rel_pos=v_output[:3]
            # print(rel_pos.shape)
            # print(robot_pos.shape)
            new_pos=robot_pos+rel_pos
            # return new_pos.cpu().data.numpy()
            waypoint_pos_rot=v_output[3:]

            o_new=torch.cat([
                robot_pos, robot_rot, jaw,
                new_pos, rel_pos, waypoint_pos_rot
            ])
            # print('o_new: ',o_new)
            o_norm=self.agent.o_norm.normalize(o_new,device=self.device)
            # print("goal ", goal)
            g_norm=self.agent.g_norm.normalize(goal, device=self.device)
            # print("g ",g)
            # g_norm=torch.tensor(g).float().to(self.device)
            input_tensor=torch.cat((o_norm, g_norm), axis=0).to(torch.float32)
            # save_input=input_tensor.cpu().data.numpy()
            # np.save('test_record/actor_input.npy',save_input)
            action = self.agent.actor(input_tensor).cpu().data.numpy().flatten()
        return action

    def get_euler_from_matrix(self, mat):
        """
        :param mat: rotation matrix (3*3)
        :return: rotation in 'xyz' euler
        """
        rot = R.from_matrix(mat)
        return rot.as_euler('xyz', degrees=False)

    def get_matrix_from_euler(self, ori):
        """
        :param ori: rotation in 'xyz' euler
        :return: rotation matrix (3*3)
        """
        rot = R.from_euler('xyz', ori)
        return rot.as_matrix()

    def wrap_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def convert_pos(self,pos,matrix):
        '''
        input: ecm pose matrix 4x4
        output rcm pose matrix 4x4
        '''
        return np.matmul(matrix[:3,:3],pos)+matrix[:3,3]

    def convert_rot(self, euler_angles, matrix):
        # Convert Euler angles to rotation matrix
        # return: matrix
        roll, pitch, yaw = euler_angles
        R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotation_matrix = np.matmul(R_z, np.matmul(R_y, R_x))

        # Invert the extrinsic matrix
        extrinsic_matrix_inv = np.linalg.inv(matrix)

        # Extract the rotation part from the inverted extrinsic matrix
        rotation_matrix_inv = extrinsic_matrix_inv[:3, :3]

        # Perform the rotation
        position_rotated = np.matmul(rotation_matrix_inv, rotation_matrix)

        return position_rotated

    def rcm2tip(self, rcm_action):
        return np.matmul(rcm_action, self.tool_T_tip)

    def _set_action(self, action, robot_pos, rot):
        '''
        robot_pos in cam coodinate
        robot_rot in rcm; matrix
        '''
        action[:3] *= 0.01 * self.scaling
        # action[1]=action[1]*-1
        # print(action)

        ecm_pos=robot_pos+action[:3]
        # print('aft robot pos tip ecm: ',ecm_pos)
        psm_pose=np.zeros((4,4))

        psm_pose[3,3]=1
        psm_pose[:3,:3]=rot
        # print('ecm pos: ',ecm_pos)
        rcm_pos=self.convert_pos(ecm_pos,basePSM_T_cam)
        # print('aft robot pos tip rcm: ',rcm_pos)
        psm_pose[:3,3]=rcm_pos

        return psm_pose

    def convert_point_to_camera_axis(self, x, y, depth, intrinsics_matrix):
        ''' 
        # Example usage
        x = 100
        y = 200
        depth = 5.0
        intrinsics_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

        xc, yc, zc = convert_point_to_camera_axis(x, y, depth, intrinsics_matrix)
        print(f"Camera axis coordinates: xc={xc}, yc={yc}, zc={zc}")
        '''
        # Extract camera intrinsics matrix components
        fx, fy, cx, cy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1], intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]

        # Normalize pixel coordinates
        xn = (x - cx) / fx
        yn = (y - cy) / fy

        # Convert to camera axis coordinates
        xc = xn * depth
        yc = yn * depth
        zc = depth

        return np.array([xc, yc, zc])

    def goal_distance(self,goal_a, goal_b):
        assert goal_a.shape==goal_b.shape
        return np.linalg.norm(goal_a-goal_b,axis=-1)

    def is_success(self, curr_pos, desired_goal):
        d=self.goal_distance(curr_pos, desired_goal)
        d3=np.abs(curr_pos[2]-desired_goal[2])
        print('distance: ',d)
        print('distance z-axis: ',d3)
        # if d3<0.007:
        #     return True
        return (d<self.threshold and d3<0.008).astype(np.float32)

    def init_run(self):
        intrinsics_matrix=np.array([[916.367081, 1.849829, 381.430393], [0.000000, 918.730361, 322.704614], [0.000000, 0.000000, 1.000000]])
        # edit for csr
        # self.p = dvrk.psm('PSM1')

        self.p= ArmProxy(node_, "psa3")
        while(not self.p.is_connected):
            self.p.measured_cp()
        # To check if the arm is connected
        self.p.read_rtrk_arm_state()
        print("connection: ",self.p.is_connected)
        # end edit for csr

        self._finished=False
        # player=VisPlayer()

        self._load_depth_model()
        # player._load_dam()

        # self._load_policy_model(filepath='./pretrained_models/s71_DDPG_demo0_traj_best.pt')
        # self._load_policy_model(filepath='./pretrained_models/csr_ar_policy.pt')
        # self._load_fastsam()

        self.cap_0=VideoCapture("/dev/video0") # left 708
        self.cap_2=VideoCapture("/dev/video2") # right 708
        print("kkkkkkkkkk")
        # init
        # open jaw
        # self.p.jaw.move_jp(np.array(-0.1)).wait()
        # print("open jaw")

        # 0. define the goal
        # TODO the goal in scaled image vs. goal in simualtor?
        for i in range(10):
            print(i)
            frame1=self.cap_0.read()
            frame2=self.cap_2.read()

        # point=SetPoints("test", frame1)

        # edit for csr camera
        self.fs = cv2.FileStorage("/home/student/csr_test/endoscope_calibration.yaml", cv2.FILE_STORAGE_READ)
        # edit for csr camera end
        print("a")
        frame1, frame2 = my_rectify(frame1, frame2, self.fs)

        print("b")
        frame1=cv2.resize(frame1, self.img_size)
        frame2=cv2.resize(frame2, self.img_size)

        print("c")
        # point=SetPoints("Goal Selection", frame1)
        # self.object_point=point[0]
        # bg_point=point[2]
        # bg_point[3]=
        print("d")
        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        print("jjjjjjjjjjjjj")
        goal= np.array( [ -0.0983023,  -0.0408574,   -0.149281])
        # trick to align the rotation with simualtor
        self.init_rotate_ecm = np.array([9.70763688e-01, 2.61508163e-05, -3.74627977e-05])
        print("dddddddddddd")
        self.rcm_goal=goal.copy()
        self.goal=self.convert_pos(goal, cam_T_basePSM)

        print("Selected Goal ecm: ",self.goal)
        print("Selected Goal rcm: ",self.rcm_goal)

        self.count=0

        self.img_data = {}
        self._thread = threading.Thread(target=self._get_image_worker)
        self.is_active = True
        self._thread.start()
        time.sleep(1)

    
    def get_center_depth(self):
        frame1 = self.cap_0.read()
        frame2 = self.cap_2.read()
        frame1, frame2 = my_rectify(frame1, frame2, self.fs)

        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        depth=self._get_depth(frame1, frame2)
        depth=cv2.resize(depth, self.img_size, interpolation=cv2.INTER_NEAREST)
        s = depth.shape
        cx = s[0] // 2
        cy = s[1] // 2
        return depth[cx][cy]

    def get_image(self):
        while True:
            if len(self.img_data)!=0:
                break
        if self._depth_remap:
            data = {k: depth_remap(v, self.depth_center, 0.3) if k!="mask" else {i: (depth_remap(j[0], self.depth_center, 0.3), 1) for i,j in v.items()} for k,v in self.img_data.items()}
        else:
            data = self.img_data
        print('get_image', data["depReal"])
        return deepcopy(data)

    def _get_image_worker(self):
        while self.is_active:
            self._update_image()
        print("exit stereo worker")

    def _update_image(self):
        frame1 = self.cap_0.read()
        frame2 = self.cap_2.read()
        frame1, frame2 = my_rectify(frame1, frame2, self.fs)

        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        depth=self._get_depth(frame1, frame2)
        depth=cv2.resize(depth, self.img_size, interpolation=cv2.INTER_NEAREST)
        frame1=cv2.resize(frame1, self.img_size)
        frame2=cv2.resize(frame2, self.img_size)


        low = self.depth_center - self.depth_range / 2
        high = self.depth_center + self.depth_range / 2
        depth_px = scale_arr(depth, low, high, 0, 255)
        depth_px = np.clip(depth_px, 0 , 255)
        depth_px = np.uint8(depth_px)

        frame1=cv2.resize(frame1, (600,600))
        depth_px=cv2.resize(depth_px, (600,600))
        d = cv2.resize(depth, (600,600))
        self.img_data["depReal"] = d.copy()
        print("in the update", self.img_data["depReal"])
        self.img_data["rgb"] = frame1
        self.img_data["depth"] = depth_px
        if self._segmentor is not None:
            self.img_data["mask"] = self._segmentor.predict(self.img_data['rgb'])
    def close(self):
        print("call stereo delelte")
        self.is_active = False
        self._thread.join()
        print("after join")
        self.cap_0.release()
        self.cap_2.release()
        print("after release")
        del self._thread


    def run_step(self):
        if self._finished:
            return True

        start_time = time.time()
        # time.sleep(.5)
        self.count+=1
        print("--------step {}----------".format(self.count))
        # time.sleep(2)

        frame1=self.cap_0.read()
        frame2=self.cap_2.read()

        # fs = cv2.FileStorage("/home/kj/ar/EndoscopeCalibration/calibration_new.yaml", cv2.FILE_STORAGE_READ)

        frame1, frame2 = my_rectify(frame1, frame2, self.fs)

        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        depth=self._get_depth(frame1, frame2)

        depth=cv2.resize(depth, self.img_size, interpolation=cv2.INTER_NEAREST)

        # print(frame1.shape)
        print('depth shape: ',depth.shape)
        # np.save('/home/kj/ar/GauzeRetrievel/test_record/depth.npy',depth)
        # print(depth[self.object_point[0]][self.object_point[1]])

        plt.imsave('test_record/pred_depth_{}.png'.format(self.count),depth)
        # exit()

        frame1=cv2.resize(frame1, self.img_size)
        frame2=cv2.resize(frame2, self.img_size)

        plt.imsave( 'test_record/frame1_{}.png'.format(self.count),frame1)
        plt.imsave( 'test_record/frame2_{}.png'.format(self.count),frame2)

        #     seg=self._seg_with_fastsam(frame1,self.object_point)
        #     #print(seg)

        #     seg=np.array(seg==True).astype(int)

        #     np.save('test_record/seg.npy',seg)
        #     plt.imsave('test_record/seg_{}.png'.format(self.count),seg)
        #     #seg=np.load('/home/kj/ar/peg_transfer/test_record/seg_from_depth.npy')
        #     print("finish seg")

        #     robot_pose=self.p.measured_cp()
        #     robot_pos=robot_pose.p
        #     print("pre action pos rcm: ",robot_pos)
        #     robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
        #     #robot_pos=player.rcm2tip(robot_pos)
        #     pre_robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
        #     # can be replaced with robot_pose.M.GetRPY()
        #     # start
        #     transform_2=robot_pose.M
        #     np_m=np.array([[transform_2[0,0], transform_2[0,1], transform_2[0,2]],
        #                         [transform_2[1,0], transform_2[1,1], transform_2[1,2]],
        #                         [transform_2[2,0], transform_2[2,1], transform_2[2,2]]])

        #     tip_psm_pose=np.zeros((4,4))

        #     tip_psm_pose[3,3]=1
        #     tip_psm_pose[:3,:3]=np_m
        #     tip_psm_pose[:3,3]=robot_pos
        #     #print('tip_psm_pose before: ',tip_psm_pose)
        #     tip_psm_pose=self.rcm2tip(tip_psm_pose)
        #     #print('tip_psm_pose aft: ',tip_psm_pose)

        #     np_m=tip_psm_pose[:3,:3]
        #     robot_pos=tip_psm_pose[:3,3]
        #     #print("pre action pos tip rcm: ",robot_pos)

        #     #robot_rot=np_m
        #     robot_rot=self.get_euler_from_matrix(np_m)
        #     robot_rot=self.convert_rot(robot_rot, cam_T_basePSM)
        #     robot_rot=self.get_euler_from_matrix(robot_rot)
        #     robot_pos=self.convert_pos(robot_pos,cam_T_basePSM)
        #     #print("pre action pos tip ecm: ",robot_pos)
        #     # end

        #     jaw=np.array([0.0]).astype(np.float64)
        #     # edit for csr
        #     # if you need the jaw value, uncomment the next line
        #     # jaw=[np.array(self.p.measured_jp())[6]-0.5]

        #     # edit for csr end

        #     robot_rot = self.init_rotate_ecm.copy()
        #     action=self._get_action(seg, depth ,robot_pos, robot_rot, jaw, self.goal)
        #     #print("finish get action")
        #     print("action: ",action)
        #     #obtained_object_position=player.convert_pos(action, basePSM_T_cam)
        #     #print('obtained_object_position: ',obtained_object_position)
        #     #PSM2_pose=PyKDL.Vector(obtained_object_position[0], obtained_object_position[1], obtained_object_position[2])

        #     # edit for csr
        #     PSM2_rotate=PyCSR.Rotation( -0.188942,   -0.979432,  -0.0708113,
        # -0.966486,    0.172712,    0.189935,
        # -0.173798,    0.104325,    -0.97924)
        #     # edit for csr end

        #     print('time:',time.time()-start_time)

        #     action_len=15
        #     action_split=action/action_len
        #     for i in range(action_len):
        #     # 4. action -> state
        #         '''
        #         robot_pose=self.p.measured_cp()
        #         robot_pos=robot_pose.p
        #         robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
        #         robot_pos=self.convert_pos(robot_pos,cam_T_basePSM)
        #         '''
        #         # print(robot_pos_new)
        #         state=self._set_action(action_split.copy(), robot_pos, np_m)

        #         #edit for csr
        #         PSM2_pose = PyCSR.Vector(state[0,-1], state[1,-1], state[2,-1])
        #         curr_robot_pos=np.array([state[0,-1], state[1,-1], state[2,-1]])

        #         print("target pos : ",curr_robot_pos)

        #         move_goal = PyCSR.Frame(PSM2_rotate, PSM2_pose)

        #         # move
        #         self.p.move_cp(move_goal, acc=1, duration=1, jaw=0)
        #         #edit for csr end

        #         #print('goal:',move_goal)
        #         #self.p.servo_cp(move_goal)
        #         if i==(action_len-1):
        #             break
        #         time.sleep(0.6)
        #         robot_pos=self.convert_pos(curr_robot_pos,cam_T_basePSM)

        #     print("finish move")
        #     print('is sccess: ',self.is_success(curr_robot_pos, self.rcm_goal))
        #     if self.is_success(curr_robot_pos, self.rcm_goal) or self.count>10:

        #         self._finished=True
        self._finished=True
        return self._finished

    def record_video(self, out1, out2):
        for i in range(10):
            frame1=self.cap_0.read()
            frame2=self.cap_2.read()
            out1.write(frame1)
            out2.write(frame2)
        return 


# import threading

if __name__=="__main__":
    #lock = threading.Lock()
    
    player=VisPlayer()
    player.init_run()
    finished=False
    while not finished:
        #player.record_video
        finished=player.run_step()


    # closing all open windows
    # cv2.destroyAllWindows()
    time.sleep(1.0)
    player.cap_0.release()
    player.cap_2.release()
    time.sleep(1.0)

    path = './test_record/frame1_1.png'

    # Reading an image in default mode
    image = cv2.imread(path)

    # Window name in which image is displayed
    window_name = 'image'

    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow(window_name, image)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
