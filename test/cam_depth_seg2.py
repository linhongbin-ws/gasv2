import cv2
from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
import numpy as np

d2=np.load("render.npy", allow_pickle=True)
keys=['rgb', 'depReal', 'mask']
img = {k: d2.item().get(k) for k in keys}
def erode_mask(mask, kernel_size=7):
    in_mask = np.zeros(mask.shape, dtype=np.uint8)
    in_mask[v] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    erode = cv2.morphologyEx(in_mask, cv2.MORPH_ERODE, kernel)
    erode_mask = erode !=0
    return erode_mask


def edge_detection(depth, segmask, depth_thres=0.001):
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
        if boundary[i] >=0:
            out_mask_left[i, boundary[i]:] = True
        else:
            out_mask_left[i,:] = segmask[i]
    
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

    plt.subplot(1,2, 1)
    imshow(right_mask)
    plt.colorbar()
    plt.subplot(1,2, 2)
    imshow(seg_depth)
    plt.colorbar()
    show()

    idx_mat1 = idx_mat[:,:-1].copy()
    s = idx_mat1.shape[1]
    idx_mat1[np.logical_not(right_mask)] = s
    boundary = np.min(idx_mat1, axis=1)
    out_mask_right = np.zeros(segmask.shape, dtype=bool)
    for i in range(boundary.shape[0]):
        arr = idx_mat[i][segmask[i]]
        if arr.shape[0] == 0:
            continue
        if boundary[i] <s:
            out_mask_right[i, : boundary[i]] = True
        else:
            out_mask_right[i,:] = segmask[i]
    
    out_mask = np.logical_and(out_mask_left, out_mask_right)

    seg_depth_out = seg_depth.copy()
    seg_depth_out[np.logical_not(out_mask)] = 0

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

        
    return seg_depth_out


seg_depth = []

for k,v in img['mask'].items():
    t = img['depReal'].copy()
    t[np.logical_not(v)] = 0
    seg_depth.append(t)

    # t1 = img['depReal'].copy()
    # v1 = erode_mask(v)
    # t1[np.logical_not(v1)] = 0
    # seg_depth.append(t1)

    new_t = edge_detection(img['depReal'].copy(), v)
    seg_depth.append(new_t)








plot_list = [img['rgb'], img['depReal']]
plot_list.extend(seg_depth)

# plot_list = [v for i, v in enumerate(plot_list) if i in [2,3]]

for n, v in enumerate(plot_list):
    plt.subplot(1,len(plot_list), n+1)
    imshow(v)
    plt.colorbar()
show()


print("exit")
# env.unwrapped.client._cam_device._device.close()



