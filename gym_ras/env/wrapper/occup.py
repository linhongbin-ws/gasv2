from gym_ras.env.wrapper.base import BaseWrapper
from gym_ras.tool.o3d import depth_image_to_point_cloud, pointclouds2occupancy
# from gym_ras.tool.depth import depth_image_to_point_cloud, pointclouds2occupancy
from gym_ras.tool.depth import occup2image
import numpy as np
import cv2
from gym_ras.tool.common import getT, invT, TxT
import time


class Occup(BaseWrapper):
    def __init__(
        self, env, mask_key=["psm1", "stuff"], is_skip=False,
        occup_h=200,
        occup_w=200,
        occup_d=200,
        pc_x_min=-0.1 * 5,
        pc_x_max=0.1 * 5,
        pc_y_min=-0.1 * 5,
        pc_y_max=0.1 * 5,
        pc_z_min=-0.1 * 5,
        pc_z_max=0.1 * 5,
        cam_offset_x=0,
        cam_offset_y=0,
        cam_offset_z=0.2 * 5,
        cam_offset_rx=45,
        cam_offset_ry=0,
        cam_offset_rz=0,
        cam_cal_file='',
        **kwargs
    ):
        super().__init__(env, **kwargs)
        self._mask_key = mask_key
        self._occup_h = occup_h
        self._occup_w = occup_w
        self._occup_d = occup_d
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
        if cam_cal_file != "":
            fs = cv2.FileStorage(cam_cal_file, cv2.FILE_STORAGE_READ)
            fn_M1 = fs.getNode("M1").mat()
            fn_M1[0][0] = fn_M1[1][1]
            fn_M1[0][2] = fn_M1[1][2]
            # print("instrinsic matrisxx: ", fn_M1)
            self._K = fn_M1
        else:
            self._K = self.unwrapped.instrinsic_K

    @property
    def occup_pc_range(self):
        return np.abs(self._pc_x_max - self._pc_x_min), \
            np.abs(self._pc_y_max - self._pc_y_min), \
            np.abs(self._pc_z_max - self._pc_z_min)

    def render(
        self,
    ):
        import time
        start = time.time()
        imgs = self.env.render()
        print(f"bottom render {time.time() - start}")
        rgb = imgs["rgb"]
        depth = imgs["depReal"]

        # get encode mask
        encode_mask = np.zeros(depth.shape, dtype=np.uint8)
        masks = [imgs["mask"][k] for k in self._mask_key]
        for m_id, m in enumerate(masks):
            encode_mask[m] = m_id + 1

        # depth image to point clouds
        scale = 1
        pose = np.eye(4)
        points = depth_image_to_point_cloud(
            rgb, depth, scale, self._K, pose, encode_mask=encode_mask, tolist=False
        )
        print(f"to point clouds {time.time() - start}")

        # transform point clouds
        if self._cam_offset_z < 0:
            s = imgs["depReal"].shape
            cx = s[0] // 2
            cy = s[1] // 2
            cam_offset_z = imgs["depReal"][cx][cy]
        else:
            cam_offset_z = self._cam_offset_z
        if self._K is None:
            self._K = self.unwrapped.instrinsic_K
        T1 = getT([-self._cam_offset_x,
                   -self._cam_offset_y,
                   -cam_offset_z,], [0, 0, 0], rot_type="euler")
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
        print(f"before occup {time.time() - start}")

        # point clouds to occupancy and images
        occup_imgs = {}
        occup_mats = {}
        for m_id, _ in enumerate(masks):
            _points = points[points[:, 6] == m_id + 1]  # mask out
            occ_mat = pointclouds2occupancy(
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
            print(f" occup {m_id} {time.time() - start}")
            occup_mats[self._mask_key[m_id]] = occ_mat
            z, z_mask = occup2image(occ_mat, image_type="depth",
                                    background_encoding=255)
            print(f" occup {m_id} image {time.time() - start}")
            z = np.uint8(255 - z)
            size = imgs["rgb"].shape[0]
            z = cv2.resize(z, (size,size), interpolation=cv2.INTER_AREA)
            z_mask = self._resize_bool(z_mask, size)
            occup_imgs[self._mask_key[m_id]] = [z, z_mask]
            # print("3", time.time()-start)
        imgs["occup_zimage"] = occup_imgs
        imgs["occup_mat"] = occup_mats
        self._occup_mat = occup_mats
        print(f"occup render time {time.time() - start}")
        return imgs

    def _resize_bool(self, im, size):
        _in = np.zeros(im.shape, dtype=np.uint8)
        _in[im] = 1
        _out = cv2.resize(_in, (size, size))
        return _out == 1

    @property
    def occup_mat(self):
        return self._occup_mat

    @property
    def occup_grid_size(self):
        return (self._pc_x_max - self._pc_x_min) / self._occup_h
