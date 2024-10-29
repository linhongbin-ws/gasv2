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
                 **kwargs,
                 ):
        super().__init__(env)
        self._skip = skip
        self._uncert_scale = uncert_scale
        self._eval = eval
        self._erode_kernel = erode_kernel

    def render(self, ):
        imgs = self.env.render()
        if not self._skip:
            if self._erode_kernel >0 and 'mask' in imgs:
                imgs['mask'] = self._erode_mask(imgs['mask'])
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
