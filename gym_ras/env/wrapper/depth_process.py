from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
from gym_ras.tool.common import scale_arr
import cv2

class DepthProcess(BaseWrapper):

    def __init__(self, env,
                 skip=True,
                 uncert_scale=1.0,
                 erode_kernel=0,
                 eval=False,
                #  edge_dectection=True,
                 edge_detect_thres=0.0,
                 **kwargs,
                 ):
        super().__init__(env)
        self._skip = skip
        self._uncert_scale = uncert_scale
        self._eval = eval
        self._erode_kernel = erode_kernel
        self._edge_detect_thres = edge_detect_thres


    def render(self, ):
        imgs = self.env.render()
        if not self._skip:
            imgs = self._process(imgs)
        return imgs

    def _process(self, imgs):
        if self._erode_kernel >0 and 'mask' in imgs:
            imgs['mask'] = self._erode_mask(imgs['mask'])
        
        if self._edge_detect_thres > 0 and 'mask' in imgs:
            for k, v in imgs['mask'].items():
                imgs['mask'][k] = self._edge_detection_proc(imgs['depReal'], v)


        imgs = self._no_depth_guess_process(imgs)
        return imgs
    
    def _erode_mask(self, masks):
        new_mask = {}
        for k,v in masks.items():
            new_mask[k] = self._erode_func(v)
        return new_mask

    def _erode_func(self, mask):
        in_mask = np.zeros(mask.shape, dtype=np.uint8)
        in_mask[mask] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._erode_kernel,self._erode_kernel))
        erode = cv2.morphologyEx(in_mask, cv2.MORPH_ERODE, kernel)
        erode_mask = erode !=0
        return erode_mask
        

    def _no_depth_guess_process(self, imgs,):
        _depth = imgs["depth"].copy()
        for target, source in self.unwrapped.nodepth_guess_map.items():
            # holes_mask = np.logical_and(_depth== 0, imgs["mask"][target])
            if target not in imgs["mask"]:
                continue
            holes_mask = imgs["mask"][target]
            if np.sum(imgs["mask"][source]) == 0:  # no mask
                continue
            _guess_val = np.median(imgs["depth"][imgs["mask"][source]])
            # print(imgs["depth"][imgs["mask"][source]])
            uncert = self.unwrapped.nodepth_guess_uncertainty[target] * \
                self._uncert_scale
            uncert_pix = scale_arr(
                uncert, 0, self.unwrapped.depth_remap_range[0][1]-self.unwrapped.depth_remap_range[0][0], 0, 255) // 2
            if not self._eval:
                _guess_val += np.random.uniform(-uncert_pix, uncert_pix)

            rand_mat = np.clip(np.random.uniform(
                _guess_val-uncert_pix, _guess_val+uncert_pix, size=imgs["depth"].shape), 0, 255)
            # print(rand_mat)
            _depth[holes_mask] = rand_mat[holes_mask]
        imgs["depth"] = np.uint8(_depth)
        return imgs


    def _edge_detection_proc(self, depth, segmask,):
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

        left_arr1 = depth[:,1:]
        left_arr2 = depth[:,:-1]
        err = np.abs(left_arr1 - left_arr2)

        for i, c in enumerate(center_idxs):
            if c == -1:
                err[i][:] = 0
            else:
                err[i][c:] = 0

        left_mask = err > self._edge_detect_thres


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
        

                
        right_arr1 = depth[:,:-1]
        right_arr2 = depth[:,1: ]
        err = np.abs(right_arr1 - right_arr2)

        for i, c in enumerate(center_idxs):
            if c == -1:
                err[i][:] = 0
            else:
                err[i][:c+1] = 0

        right_mask = err > self._edge_detect_thres


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
            
        return out_mask