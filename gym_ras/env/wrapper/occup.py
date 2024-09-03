from gym_ras.env.wrapper.base import BaseWrapper
from gym_ras.tool.depth import get_intrinsic_matrix, depth_image_to_point_cloud, pointclouds2occupancy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from gym_ras.tool.common import getT, invT, TxT

class Occup(BaseWrapper):
    def __init__(self, env, 
                 
    **kwargs):
        super().__init__(env, **kwargs)
    def render(self, ):
        imgs = self.env.render()
        # print(imgs.keys())
        # print(imgs['mask'].keys())
        rgb = imgs["rgb"]
        depth = imgs["depReal"]
        mask = np.logical_or(imgs['mask']['psm1'],imgs['mask']['stuff'] )
        encode_mask = np.zeros(mask.shape)
        encode_mask[mask] = 1
        scale = 1
        K = get_intrinsic_matrix(depth.shape[0],depth.shape[1],fov=45) 
        pose = np.eye(4)
        points = depth_image_to_point_cloud(rgb, depth, scale, K, pose, encode_mask = encode_mask)

        # save_ply_name = "test_pc.ply"
        # save_ply_path = Path('./data/test_pc') 
        # save_ply_path.mkdir(parents=True, exist_ok=True)
        # write_point_cloud(str(save_ply_path / save_ply_name), points)

        points = np.array(points)
        points = points[points[:,6]==1] # mask out
        T1 = getT([0,0,-0.2*5], [0,0,0], rot_type='euler')
        T2 = getT([0,0,0], [-45,0,0], rot_type='euler', euler_Degrees=True)
        ones = np.ones((points.shape[0],1))
        P = np.concatenate((points[:,:3], ones), axis=1)
        points[:,:3] = np.matmul(P, np.transpose(TxT([T2, T1,])))[:,:3]
        print(points[:,:3])
        occ_mat_image_x = pointclouds2occupancy(points, occup_h=64, occup_w=64, occup_d=64, 
                            pc_x_min=-0.1*5,pc_x_max=0.1*5,
                            pc_y_min=-0.1*5,pc_y_max=0.1*5,
                            pc_z_min=-0.1*5,pc_z_max=0.1*5)
        imgs["occ_x"]
        