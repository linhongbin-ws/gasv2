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
        self, env, mask_key=["psm1", "stuff"], fov=45, is_skip=False, **kwargs
    ):
        super().__init__(env, **kwargs)
        self._mask_key = mask_key
        self._fov = fov

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
        K = get_intrinsic_matrix(depth.shape[0], depth.shape[1], fov=self._fov)
        pose = np.eye(4)

        points = depth_image_to_point_cloud(
            rgb, depth, scale, K, pose, encode_mask=encode_mask,tolist=False
        )
        T1 = getT([0, 0, -0.2 * 5], [0, 0, 0], rot_type="euler")
        T2 = getT([0, 0, 0], [-45, 0, 0], rot_type="euler", euler_Degrees=True)
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
        occup_imgs = {}
        for m_id, _ in enumerate(masks):
            _points = points[points[:, 6] == m_id + 1]  # mask out
            (x, y, z) = pointclouds2occupancy(
                _points,
                occup_h=200,
                occup_w=200,
                occup_d=200,
                pc_x_min=-0.1 * 5,
                pc_x_max=0.1 * 5,
                pc_y_min=-0.1 * 5,
                pc_y_max=0.1 * 5,
                pc_z_min=-0.1 * 5,
                pc_z_max=0.1 * 5,
            )
            x = self._resize_bool(x, imgs['rgb'].shape[0])
            y = self._resize_bool(y, imgs["rgb"].shape[0])
            z = self._resize_bool(z, imgs["rgb"].shape[0])
            occup_imgs[self._mask_key[m_id]] = [x,y,z]
        imgs["occup"] = occup_imgs
        self._occup_projection = occup_imgs
        return imgs
    def _resize_bool(self, im, size):
        _in = np.zeros(im.shape,dtype=np.uint8)
        _in[im] = 1
        _out = cv2.resize(_in, (size, size))
        return _out == 1

    @property
    def occupancy_projection(self):
        return self._occup_projection 