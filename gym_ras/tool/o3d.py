
import open3d as o3d
import numpy as np

def depth_image_to_point_cloud( rgb, depth_real, scale, new_K, pose, encode_mask=None, tolist=False):
    width = depth_real.shape[0]
    height = depth_real.shape[1]
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=width, height=height, fx=new_K[0][0], fy=new_K[1][1], cx=new_K[0][2], cy=new_K[1][2])
    # intrinsic = o3d.core.Tensor(K[:3][:3])
    # print(np.max(depth),np.min(depth))

    scale_depth = 255
    depth_image = depth_real*scale_depth
    depth_image = depth_image.astype(np.uint16)
    color_raw = rgb.copy()
    if encode_mask is not None:
        color_raw[:,:,2] = encode_mask.astype(np.uint8)
        # print("color_raw[:,:,2]",color_raw[:,:,2])

    color_raw = np.asarray(color_raw, order="C")

    color_raw = o3d.geometry.Image(color_raw)
    depth_raw = o3d.geometry.Image(depth_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=scale_depth, depth_trunc=10000000,
                                                               convert_rgb_to_intensity=False)

    pointSet = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, 
                                            intrinsic,project_valid_depth_only=False)
    
    # pointSet,_  = pointSet.remove_statistical_outlier(nb_neighbors=20,
    #                                                 std_ratio=2.0)


    points = np.asarray(pointSet.points)
    colors = np.asarray(pointSet.colors)
    mat = np.concatenate((points,colors), axis=1)
    if encode_mask is not None:
        mat = np.concatenate((mat, mat[:,5:6]*255), axis=1)
    if tolist:
        mat = mat.tolist()
    return mat

def pointclouds2occupancy(pc_mat,
                        occup_h,
                        occup_w,
                        occup_d,
                        pc_x_min,
                        pc_x_max,
                        pc_y_min,
                        pc_y_max,
                        pc_z_min,
                        pc_z_max,
                        outlier_rm_pts=0,
                        outlier_rm_radius=0,
                        debug=False,
                          ):
    range_ = pc_x_max - pc_x_min
    resolution = occup_h
    pc_x_max = pc_x_min + range_
    pc_y_max = pc_y_min + range_
    pc_z_max = pc_z_min + range_
    min_bound = np.array([pc_x_min,pc_y_min,pc_z_min]).reshape(3,1)
    max_bound = np.array([pc_x_max,pc_y_max,pc_z_max]).reshape(3,1)


    voxel_size = range_ / resolution
    _pc_mat = pc_mat.copy()
    _pc_mat = _pc_mat[_pc_mat[:,0] >= pc_x_min]
    _pc_mat = _pc_mat[_pc_mat[:,1] >= pc_y_min]
    _pc_mat = _pc_mat[_pc_mat[:,2] >= pc_z_min]
    _pc_mat = _pc_mat[_pc_mat[:,0] <= pc_x_max]
    _pc_mat = _pc_mat[_pc_mat[:,1] <= pc_y_max]
    _pc_mat = _pc_mat[_pc_mat[:,2] <= pc_z_max]
    # print(pc_x_min,pc_y_min,pc_z_min,pc_x_max,pc_y_max,pc_z_max)
    # print(_pc_mat.shape)

    pointSet = o3d.geometry.PointCloud()
    pointSet.points = o3d.utility.Vector3dVector(_pc_mat[:,:3])
    pointSet.colors = o3d.utility.Vector3dVector(_pc_mat[:,3:6])

    if outlier_rm_pts>0 and outlier_rm_radius>0:
        # print(f"before point nums: {np.asarray(pointSet.points).shape}")
        _,ind  =pointSet.remove_radius_outlier(nb_points=outlier_rm_pts, radius=outlier_rm_radius)
        pointSet = pointSet.select_by_index(ind)
        # print(f"after point nums: {np.asarray(pointSet.points).shape}")

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pointSet, 
                                          voxel_size, 
                                          min_bound, 
                                          max_bound)
    if debug:
        o3d.visualization.draw_geometries([voxel_grid])
    vx = voxel_grid.get_voxels()
    # print(vx)
    vx_idx = [v.grid_index for v in vx]
    occ_mat = np.zeros((resolution, resolution, resolution), dtype=bool)
    # print(len(vx_idx))
    for i in vx_idx:
        # print(i)
        occ_mat[i[0],i[1],i[2]] = True
    return occ_mat

def convert_occup_mat_o3dvoxels(
        occ_mat_dict,
        pc_x_min,
        pc_x_max,
        pc_y_min,
        pc_y_max,
        pc_z_min,
        pc_z_max,
        voxel_size,
):
    grids = {}
    for k, _occ_mat in occ_mat_dict.items():
        min_bound = np.array([pc_x_min,pc_y_min,pc_z_min]).reshape(3,1)
        max_bound = np.array([pc_x_max,pc_y_max,pc_z_max]).reshape(3,1)
        pointSet = o3d.geometry.PointCloud()
        xs, ys, zs = _occ_mat.nonzero()
        origin = np.array([pc_x_min,pc_y_min,pc_z_min]).reshape(3,1)
        pc_mat = np.array([xs, ys, zs]) * voxel_size + origin
        pc_mat = np.transpose(pc_mat)
        # print(pc_mat.shape)
        pointSet.points = o3d.utility.Vector3dVector(pc_mat)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pointSet, 
                                            voxel_size, 
                                            min_bound, 
                                            max_bound)
        grids[k] = voxel_grid
    

    return grids

class O3DVis():
    def __init__(self, center_list) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.objs = []
        self.center = np.array(center_list).reshape(3,1)
    def draw(self, geometrys):
        for o in self.objs:
            self.vis.remove_geometry(o)
        for g in geometrys:
            self.vis.add_geometry(g)
            self.objs.append(g)
        ctr = self.vis.get_view_control()
        ctr.set_lookat(self.center)
        ctr.set_front(np.array([0,0,1]))
        ctr.set_up(np.array([-1,0,0]))
        ctr.set_zoom(4)
        self.vis.poll_events()
        self.vis.update_renderer()


    def __del__(self):
        self.vis.destroy_window()
        

