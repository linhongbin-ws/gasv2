# https://github.com/vitalemonate/depth2Cloud/blob/main/depth2Cloud.py
import os
import numpy as np
import cv2
from tqdm import tqdm
from gym_ras.tool.common import scale_arr
import time





def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()


def depth_image_to_point_cloud(rgb, depth, scale, K, pose, encode_mask=None, tolist=True):
    start = time.time()

    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])
    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)
    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)
    valid = Z > 0
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    stacklist = (X, Y, Z, np.ones(len(X)))
    position = np.vstack(stacklist)
    position = np.dot(pose, position)
    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]
    if encode_mask is None:
        points = np.transpose(np.vstack((position[0:3, :], R, G, B)))
    else:
        M = np.ravel(encode_mask)[valid]
        points = np.transpose(np.vstack((position[0:3, :], R, G, B, M)))
    if tolist:
        points = points.tolist()
    return points


# image_files: XXXXXX.png (RGB, 24-bit, PNG)
# depth_files: XXXXXX.png (16-bit, PNG)
# poses: camera-to-world, 4×4 matrix in homogeneous coordinates
def build_point_cloud(dataset_path, scale, view_ply_in_world_coordinate):
    K = np.fromfile(os.path.join(dataset_path, "K.txt"), dtype=float, sep="\n ")
    K = np.reshape(K, newshape=(3, 3))
    image_files = sorted(Path(os.path.join(dataset_path, "images")).files('*.png'))
    depth_files = sorted(Path(os.path.join(dataset_path, "depth_maps")).files('*.png'))

    if view_ply_in_world_coordinate:
        poses = np.fromfile(os.path.join(dataset_path, "poses.txt"), dtype=float, sep="\n ")
        poses = np.reshape(poses, newshape=(-1, 4, 4))
    else:
        poses = np.eye(4)

    for i in tqdm(range(0, len(image_files))):
        image_file = image_files[i]
        depth_file = depth_files[i]

        rgb = cv2.imread(image_file)
        depth = cv2.imread(depth_file, -1).astype(np.uint16)

        if view_ply_in_world_coordinate:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses[i])
        else:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses)
        save_ply_name = os.path.basename(os.path.splitext(image_files[i])[0]) + ".ply"
        save_ply_path = os.path.join(dataset_path, "point_clouds")

        if not os.path.exists(save_ply_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.mkdir(save_ply_path)
        write_point_cloud(os.path.join(save_ply_path, save_ply_name), current_points_3D)

def get_intrinsic_matrix(width, height, fov):
    """ Calculate the camera intrinsic matrix.
    """
    fy = fx = (width / 2.) / np.tan(fov / 2.)  # fy = fy?
    cx, cy = width / 2., height / 2.
    mat = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]]).astype(np.float)
    return mat

def pointclouds2occupancy(pc_mat, occup_h, occup_w, occup_d, 
                          pc_x_min,pc_x_max,
                          pc_y_min,pc_y_max,
                          pc_z_min,pc_z_max,
                          ):
    _pc_mat = pc_mat.copy()
    # print(_pc_mat[:,0],  pc_x_min, pc_x_max, 0, occup_h-1,)
    get_idx = lambda mat, min, max, occ_s: np.clip(scale_arr(mat, min, max, 0, occ_s-1), 0, occ_s-1).astype(int)
    # idx_x = scale_arr(_pc_mat[:,0],  pc_x_min, pc_x_max, 0, occup_h-1,)
    # idx_y = scale_arr(_pc_mat[:,1],  pc_y_min, pc_y_max, 0, occup_w-1,)
    # idx_z = scale_arr(_pc_mat[:,2],  pc_z_min, pc_z_max, 0, occup_d-1,)
    idx_x = get_idx(_pc_mat[:,0],  pc_x_min, pc_x_max,occup_h)
    idx_y = get_idx(_pc_mat[:,1],  pc_y_min, pc_y_max,occup_w)
    idx_z = get_idx(_pc_mat[:,2],  pc_z_min, pc_z_max,occup_d)
    occ_mat = np.zeros((occup_h, occup_w, occup_d), dtype=bool)
    occ_mat[idx_x,idx_y,idx_z] = True
    occ_proj_y = np.sum(occ_mat, axis=0) !=0
    occ_proj_z = np.flip(np.transpose(np.sum(occ_mat, axis=1) != 0), 0)
    occ_proj_x = np.transpose(np.sum(occ_mat, axis=2) != 0)

    return occ_proj_x, occ_proj_y, occ_proj_z





def edge_detection(depth, segmask, depth_thres=0.003, debug=False):
    seg_depth = depth.copy()
    seg_depth[np.logical_not(segmask)] = 0
    idx_mat = np.stack([np.arange(seg_depth.shape[1])]*seg_depth.shape[0], axis=0)
    center_idxs = []
    for i in range(segmask.shape[0]):
        arr = idx_mat[i][segmask[i]]
        if arr.shape[0] == 0:
            c = -1
        else:
            c = idx_mat[i][segmask[i]].mean()
            c = np.int(c)
        center_idxs.append(c)
    
    convex_rows = []
    for i in range(segmask.shape[0]):
        cvx_cnt = 0
        for j in range(segmask.shape[1]-1):
            if (segmask[i][j+1]) and (not segmask[i][j]):
                cvx_cnt+=1
        convex_rows.append(cvx_cnt==1)
            
    # center_idxs[np.logical_not(segmask)] = 0 
    # center_idx = np.int(center_idxs[segmask].mean())
    left_arr1 = depth[:,1:]
    left_arr2 = depth[:,:-1]
    err = np.abs(left_arr1 - left_arr2)
    # imshow(err)
    # plt.colorbar()
    # show()
    for i, c in enumerate(center_idxs):
        if c == -1:
            err[i][:] = 0
        else:
            err[i][c:] = 0
    # imshow(err)
    # plt.colorbar()
    # show()
    left_mask = err > depth_thres

    # plt.subplot(1,2, 1)
    # imshow(left_mask)
    # plt.colorbar()
    # plt.subplot(1,2, 2)
    # imshow(seg_depth)
    # plt.colorbar()
    # show()

    idx_mat1 = idx_mat[:,1:].copy()
    idx_mat1[np.logical_not(left_mask)] = -1
    boundary = np.max(idx_mat1, axis=1)
    out_mask_left = np.zeros(segmask.shape, dtype=bool)
    for i in range(boundary.shape[0]):
        arr = idx_mat[i][segmask[i]]
        if arr.shape[0] == 0:
            continue
        if boundary[i] >=0 and convex_rows[i]:
            out_mask_left[i, boundary[i]:] = True
        else:
            out_mask_left[i,:] = segmask[i]
    out_mask_left[np.logical_not(segmask)] = False
    
    # plt.subplot(1,2, 1)
    # imshow(out_mask_left)
    # plt.colorbar()
    # plt.subplot(1,2, 2)
    # imshow(seg_depth)
    # plt.colorbar()
    # show()
            
    right_arr1 = depth[:,:-1]
    right_arr2 = depth[:,1: ]
    err = np.abs(right_arr1 - right_arr2)
    # imshow(err)
    # plt.colorbar()
    # show()
    for i, c in enumerate(center_idxs):
        if c == -1:
            err[i][:] = 0
        else:
            err[i][:c+1] = 0
    # imshow(err)
    # plt.colorbar()
    # show()
    right_mask = err > depth_thres

    # plt.subplot(1,2, 1)
    # imshow(right_mask)
    # plt.colorbar()
    # plt.subplot(1,2, 2)
    # imshow(seg_depth)
    # plt.colorbar()
    # show()

    idx_mat1 = idx_mat[:,:-1].copy()
    s = idx_mat1.shape[1]
    idx_mat1[np.logical_not(right_mask)] = s
    boundary = np.min(idx_mat1, axis=1)
    out_mask_right = np.zeros(segmask.shape, dtype=bool)
    for i in range(boundary.shape[0]):
        arr = idx_mat[i][segmask[i]]
        if arr.shape[0] == 0:
            continue
        if boundary[i] <s and convex_rows[i]:
            out_mask_right[i, : boundary[i]] = True
        else:
            out_mask_right[i,:] = segmask[i]
    out_mask_right[np.logical_not(segmask)] = False
    
    out_mask = np.logical_and(out_mask_left, out_mask_right)

    seg_depth_out = seg_depth.copy()
    seg_depth_out[np.logical_not(out_mask)] = 0

    if debug:
        from matplotlib.pyplot import imshow, subplot, axis, cm, show
        import matplotlib.pyplot as plt
        plt.subplot(1,5, 1)
        imshow(out_mask_left)
        plt.colorbar()
        plt.subplot(1,5, 2)
        imshow(out_mask_right)
        plt.colorbar()
        plt.subplot(1,5, 3)
        imshow(out_mask)
        plt.colorbar()
        plt.subplot(1,5, 4)
        imshow(seg_depth)
        plt.colorbar()
        plt.subplot(1,5, 5)
        imshow(seg_depth_out)
        plt.colorbar()
        show()

        
    return out_mask