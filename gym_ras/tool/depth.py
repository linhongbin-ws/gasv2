# https://github.com/vitalemonate/depth2Cloud/blob/main/depth2Cloud.py
import os
import numpy as np
import cv2
from tqdm import tqdm
from gym_ras.tool.common import scale_arr
import time




def depth_image_to_point_cloud(rgb, depth, scale, K, pose, encode_mask=None, tolist=True):

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
    valid = Z > -10000
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
                          pc_x_min=None,pc_x_max=None,
                          pc_y_min=None,pc_y_max=None,
                          pc_z_min=None,pc_z_max=None,
                          ):
    _pc_mat = pc_mat.copy()
    # print(_pc_mat[:,0],  pc_x_min, pc_x_max, 0, occup_h-1,)
    get_idx = lambda mat, min, max, occ_s: np.clip(scale_arr(mat, min, max, 0, occ_s-1), 0, occ_s-1).astype(int)
    # idx_x = scale_arr(_pc_mat[:,0],  pc_x_min, pc_x_max, 0, occup_h-1,)
    # idx_y = scale_arr(_pc_mat[:,1],  pc_y_min, pc_y_max, 0, occup_w-1,)
    # idx_z = scale_arr(_pc_mat[:,2],  pc_z_min, pc_z_max, 0, occup_d-1,)
    # print(_pc_mat.shape)
    if pc_x_min is None: pc_x_min=np.min(_pc_mat[:,0])
    if pc_x_max is None: pc_x_max=np.max(_pc_mat[:,0])
    if pc_y_min is None: pc_y_min=np.min(_pc_mat[:,1])
    if pc_y_max is None: pc_y_max=np.max(_pc_mat[:,1])
    if pc_z_min is None: pc_z_min=np.min(_pc_mat[:,2])
    if pc_z_max is None: pc_z_max=np.max(_pc_mat[:,2])
    idx_x = get_idx(_pc_mat[:,0],  pc_x_min, pc_x_max,occup_h)
    idx_y = get_idx(_pc_mat[:,1],  pc_y_min, pc_y_max,occup_w)
    idx_z = get_idx(_pc_mat[:,2],  pc_z_min, pc_z_max,occup_d)
    occ_mat = np.zeros((occup_h, occup_w, occup_d), dtype=bool)
    occ_mat[idx_x,idx_y,idx_z] = True
    return occ_mat


def projection_matrix_to_K(p, image_size):
    K = np.zeros((3,3))
    K[0][0] = p[0][0] * image_size /2
    K[1][1] = p[1][1] * image_size /2
    K[0][2] =image_size /2
    K[1][2] =image_size /2
    K[2][2] = 1
    return K

def scale_K(K, width_scale, height_scale):
    _K = K.copy()
    _K[0,0] = _K[0,0]* width_scale
    _K[0,2] = _K[0,2] * width_scale
    _K[1,1] = _K[1,1] * height_scale
    _K[1,2] = _K[1,2] * height_scale
    return _K

def occup2image(occ_mat, image_type='projection', background_encoding=255
                ):
    if image_type=="projection":
        occ_proj_y = np.sum(occ_mat, axis=0) !=0
        occ_proj_z = np.flip(np.transpose(np.sum(occ_mat, axis=1) != 0), 0)
        occ_proj_x = np.transpose(np.sum(occ_mat, axis=2) != 0)
        return occ_proj_x, occ_proj_y, occ_proj_z
    elif image_type=="depth_xyz":
        s = occ_mat.shape
        index_arr = np.tile(np.arange(s[0]).reshape(-1,1,1,), (1,s[1],s[2]))
        index_arr[np.logical_not(occ_mat)] = s[0]
        x = np.min(index_arr, axis=0)
        x[x== s[0]] = 0 
        x = np.clip(scale_arr(x, 0, s[0]-1, 0, 255),0,255).astype(np.uint8)
        index_arr = np.tile(np.arange(s[1]).reshape(1,-1,1,), (s[0],1,s[2]))
        index_arr[np.logical_not(occ_mat)] = s[1]
        y = np.min(index_arr, axis=1)
        y[y== s[1]] = 0 
        y = np.clip(scale_arr(y, 0, s[1]-1, 0, 255),0,255).astype(np.uint8)
        index_arr = np.tile(np.arange(s[2]).reshape(1,1,-1,), (s[0],s[1],1))
        index_arr[np.logical_not(occ_mat)] = s[2]
        z = np.min(index_arr, axis=2)
        z[z== s[2]] = 0 
        z = np.clip(scale_arr(z, 0, s[2]-1, 0, 255),0,255).astype(np.uint8)
        z = np.transpose(z)
    elif image_type=="depth":
        s = occ_mat.shape
        index_arr = np.tile(np.arange(s[2]).reshape(1,1,-1,), (s[0],s[1],1))
        index_arr[np.logical_not(occ_mat)] = s[2]
        z = np.min(index_arr, axis=2)
        z[z== s[2]] = background_encoding 
        z = np.clip(scale_arr(z, 0, s[2]-1, 0, 255),0,255).astype(np.uint8)
        # z = np.transpose(z)
        return z, z!= background_encoding
    elif image_type=="depR":
        s = occ_mat.shape
        index_arr = np.tile(np.arange(s[2]).reshape(1,1,-1,), (s[0],s[1],1))
        index_arr[np.logical_not(occ_mat)] = s[2]
        z = np.min(index_arr, axis=2)
        backgrond_mask = z== s[2]
        z[backgrond_mask] = background_encoding 
        z = z.astype(float)
        z = np.clip(scale_arr(z, 0, s[2]-1, 0, 1),0,1)
        # z = np.transpose(z)
        return z, np.logical_not(backgrond_mask)
    else:
        raise NotImplementedError

def edge_detection(depth, segmask, depth_thres=0.003, debug=False):
    print(f"depth_thres: {depth_thres}")
    _depth = depth.copy()
    seg_depth = depth.copy()
    seg_depth[np.logical_not(segmask)] = 0

    # find center index in the mask
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
    
    # find convex index in the mask
    convex_rows = []
    seg_rows = []
    for i in range(segmask.shape[0]):
        cvx_cnt = 0
        for j in range(segmask.shape[1]-1):
            if (segmask[i][j+1]) and (not segmask[i][j]):
                cvx_cnt+=1
        convex_rows.append(cvx_cnt==1)
        seg_rows.append(cvx_cnt>=1)

    # create left mask
    left_arr1 = depth[:,1:]
    left_arr2 = depth[:,:-1]
    err = np.abs(left_arr1 - left_arr2)
    for i, c in enumerate(center_idxs):
        if c == -1:
            err[i][:] = 0
        else:
            err[i][c:] = 0
    left_mask = err > depth_thres
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
    
    # create right mask
    right_arr1 = depth[:,:-1]
    right_arr2 = depth[:,1: ]
    err = np.abs(right_arr1 - right_arr2)
    for i, c in enumerate(center_idxs):
        if c == -1:
            err[i][:] = 0
        else:
            err[i][:c+1] = 0
    right_mask = err > depth_thres
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
    
    # merge left and right masks
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
    return _depth, out_mask