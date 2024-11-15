from gym_ras.env.wrapper.base import BaseWrapper
from gym_ras.tool.depth import (
    get_intrinsic_matrix,
    depth_image_to_point_cloud,
    pointclouds2occupancy,
)
import numpy as np
import cv2
import matplotlib.pyplot as plt
from gym_ras.tool.common import getT, invT, TxT
import time


class Occup(BaseWrapper):
    def __init__(
        self, env, mask_key=["psm1", "stuff"], fov=45, is_skip=False, 
        occup_h=200,
        occup_w=200,
        occup_d=200,
        pc_x_min=-0.1 * 5,
        pc_x_max=0.1 * 5,
        pc_y_min=-0.1 * 5,
        pc_y_max=0.1 * 5,
        pc_z_min=-0.1 * 5,
        pc_z_max=0.1 * 5,    
        cam_offset_x = 0,
        cam_offset_y = 0,
        cam_offset_z = 0.2 * 5,
        cam_offset_rx = 45,
        cam_offset_ry = 0,
        cam_offset_rz = 0,     
        cam_cal_file = '',   
        cam_fov = 45,    
        **kwargs
    ):
        super().__init__(env, **kwargs)
        self._mask_key = mask_key
        self._fov = fov
        self._occup_h=occup_h
        self._occup_w=occup_w
        self._occup_d=occup_d
        self._pc_x_min = pc_x_min
        self._pc_x_max = pc_x_max
        self._pc_y_min = pc_y_min
        self._pc_y_max = pc_y_max
        self._pc_z_min = pc_z_min
        self._pc_z_max = pc_z_max
        self._cam_offset_x = cam_offset_x
        self._cam_offset_y = cam_offset_y
        self._cam_offset_z = cam_offset_z 
        self._cam_offset_rx = cam_offset_rx
        self._cam_offset_ry = cam_offset_ry
        self._cam_offset_rz = cam_offset_rz
        self._cam_cal_file = cam_cal_file
        self._cam_fov = cam_fov

        if cam_cal_file != "":
            fs = cv2.FileStorage(cam_cal_file, cv2.FILE_STORAGE_READ)
            fn_M1= fs.getNode("M1").mat()
            fn_M1[0][0] = fn_M1[1][1]
            fn_M1[0][2] = fn_M1[1][2]
            # print("instrinsic matrisxx: ", fn_M1)
            self._K = fn_M1
        else:
            self._K = None
    
    @property
    def occup_pc_range(self):
        return np.abs(self._pc_x_max - self._pc_x_min), \
                np.abs(self._pc_y_max - self._pc_y_min), \
                np.abs(self._pc_z_max - self._pc_z_min)

    def render(
        self,
    ):
        imgs = self.env.render()
        start = time.time()

        rgb = imgs["rgb"]
        depth = imgs["depReal"]
        encode_mask = np.zeros(depth.shape, dtype=np.uint8)
        masks = [imgs["mask"][k] for k in self._mask_key]
        for m_id, m in enumerate(masks):
            encode_mask[m] = m_id + 1
        scale = 1

        if self._cam_offset_z < 0 :
            s = imgs["depReal"].shape
            cx = s[0] // 2
            cy = s[1] // 2
            cam_offset_z = imgs["depReal"][cx][cy]

        else:
            cam_offset_z = self._cam_offset_z
        # print(cam_offset_z, "cam_offset_z")

        if self._K is None:
            self._K = get_intrinsic_matrix(depth.shape[0], depth.shape[1], fov=self._cam_fov)
        pose = np.eye(4)
        # print(self._K)
        points = depth_image_to_point_cloud(
            rgb, depth, scale, self._K, pose, encode_mask=encode_mask, tolist=False
        )
        T1 = getT([-self._cam_offset_x,
                   -self._cam_offset_y,
                   -cam_offset_z,], [0,0,0], rot_type="euler")
        T2 = getT([0, 0, 0],
                   [-self._cam_offset_rx,
                   -self._cam_offset_ry,
                   -self._cam_offset_rz,], rot_type="euler", euler_Degrees=True)
        ones = np.ones((points.shape[0], 1))
        P = np.concatenate((points[:, :3], ones), axis=1)
        points[:, :3] = np.matmul(
            P,
            np.transpose(
                TxT(
                    [
                        T2,
                        T1,
                    ]
                )
            ),
        )[:, :3]
        # print("1.2", time.time()-start)
        occup_imgs = {}
        # print("2", time.time()-start)
        for m_id, _ in enumerate(masks):
            _points = points[points[:, 6] == m_id + 1]  # mask out
            (x, y, z) = pointclouds2occupancy(
                _points,
                occup_h=self._occup_h,
                occup_w=self._occup_w,
                occup_d=self._occup_d,
                pc_x_min=self._pc_x_min,
                pc_x_max=self._pc_x_max,
                pc_y_min=self._pc_y_min,
                pc_y_max=self._pc_y_max,
                pc_z_min=self._pc_z_min,
                pc_z_max=self._pc_z_max,
            )
            x = self._resize_bool(x, imgs["rgb"].shape[0])
            y = self._resize_bool(y, imgs["rgb"].shape[0])
            z = self._resize_bool(z, imgs["rgb"].shape[0])
            occup_imgs[self._mask_key[m_id]] = [x, y, z]
            # print("3", time.time()-start)
        imgs["occup"] = occup_imgs
        self._occup_projection = occup_imgs
        return imgs

    def _resize_bool(self, im, size):
        _in = np.zeros(im.shape, dtype=np.uint8)
        _in[im] = 1
        _out = cv2.resize(_in, (size, size))
        return _out == 1

    @property
    def occupancy_projection(self):
        return self._occup_projection
