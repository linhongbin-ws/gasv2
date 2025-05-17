from gym_ras.tool.track_any_tool import Tracker
from gym_ras.api import make_env
from gym_ras.env.wrapper import Visualizer, ActionOracle
import numpy as np
import argparse
from tqdm import tqdm
import time
from copy import deepcopy
parser = argparse.ArgumentParser()
parser.add_argument('--reset', action="store_true")
args = parser.parse_args()

env, env_config = make_env(tags=['gasv2_dvrk'], seed=0)

if args.reset:
    env.unwrapped.client.psm_reset_pose()

print("start tracker....")

class NewTracker(Tracker):
    def vos_tracking_video(self, video_state, interactive_state, mask_dropdown):
        video_state_origin = deepcopy(video_state)
        interactive_state_origin = deepcopy(interactive_state)
        # print(video_state)
        # print(interactive_state)
        # print(mask_dropdown)
        if interactive_state["multi_mask"]["masks"]:
            if len(mask_dropdown) == 0:
                mask_dropdown = ["mask_001"]
            mask_dropdown.sort()
            template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
            for i in range(1,len(mask_dropdown)):
                mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
                template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
            video_state["masks"][video_state["select_frame_number"]]= template_mask
        else:      
            template_mask = video_state["masks"][video_state["select_frame_number"]]
        

        gt_mask = template_mask

    
        pred_mask = self.track(self.ref_frames[0])
        intersection = np.logical_and(pred_mask==1, gt_mask==1, dtype=np.bool )
        union = np.logical_or(pred_mask==1, gt_mask==1, dtype=np.bool )
        iou1 = np.sum(intersection) /  np.sum(union)

        intersection = np.logical_and(pred_mask==2, gt_mask==2, dtype=np.bool )
        union = np.logical_or(pred_mask==2, gt_mask==2, dtype=np.bool )
        iou2 = np.sum(intersection) /  np.sum(union)
        print(f"IOU: ")
        print(f"[{iou1}, {iou2}],")
        # mask == 1
        import cv2
        new_img = np.zeros(template_mask.shape, dtype=np.uint8)
        new_img[gt_mask==1] = 255
        new_img[gt_mask==2] = 255
        new_img[gt_mask==3] = 255
        cv2.imwrite("./debug_mask_gt.png",new_img)
        new_img = np.zeros(pred_mask.shape, dtype=np.uint8)
        new_img[pred_mask==1] = 255
        new_img[pred_mask==2] = 255
        new_img[pred_mask==3] = 255
        cv2.imwrite("./debug_mask_pred.png",new_img)
        # cv2.imwrite("./debug_rgb.png",self.ref_frames)
        
        # imgplot = plt.imshow(self.ref_frames)
        # plt.show()

        
        return None, video_state_origin, interactive_state_origin, None

    
    def inpaint_video(self, video_state, interactive_state, mask_dropdown):
        img = env.client._cam_device._device.get_image()
        self.set_rgb_image(img['rgb'])
        return None, None, None , None


            
tracker = NewTracker(sam=True, xmem=True)
tracker.load_template()
tracker.update_template()
tracker.reset()
img = env.client._cam_device._device.get_image()
tracker.set_rgb_image(img['rgb'])
tracker.gui()
