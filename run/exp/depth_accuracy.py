import numpy as np
import cv2 
from gym_ras.tool.stereo_dvrk import VisPlayer
from gym_ras.tool.o3d import depth_image_to_point_cloud, pointclouds2occupancy

cam_cal_file = "./config/cam_cal.yaml"

fs = cv2.FileStorage(cam_cal_file, cv2.FILE_STORAGE_READ)
fn_M1 = fs.getNode("M1").mat()
# fn_M1[0][0] = fn_M1[1][1]
# fn_M1[0][2] = fn_M1[1][2]

# fn_M1[1][1] = fn_M1[0][0]
# fn_M1[1][2] = fn_M1[0][2] 
# print("instrinsic matrisxx: ", fn_M1)
_K = np.zeros((3,3))
_K[1][1] = fn_M1[0][0]
_K[0][0] = fn_M1[1][1]
_K[0][2] = 300
_K[1][2] = 300
# self._K = self._K/10
print(f"read cam_cal_file {cam_cal_file}, K: {_K}")

engine = VisPlayer()
engine.init_run()

collect_two_points = []

distances = []





def onMouse(event, x, y, flags, param):
    global collect_two_points
    global distances
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw circle here (etc...)
        # print('x = %d, y = %d'%(x, y))

        encode_idx = 7
        rgb = imgs["rgb"]
        depth = imgs["depReal"]

        # get encode mask
        encode_mask = np.zeros(depth.shape, dtype=np.uint8)
        encode_mask[y, x] = encode_idx

        # depth image to point clouds
        scale = 1
        pose = np.eye(4)
        points = depth_image_to_point_cloud(
            rgb, depth, scale, _K, pose, encode_mask=encode_mask, tolist=False
        )

        _points = points[points[:, 6] == encode_idx]  # mask out
        collect_two_points.append(_points[0,:3])
        print(f"points:      {_points[0,:3]}")

        if len(collect_two_points) >=2:
            d  = np.linalg.norm(collect_two_points[0]-collect_two_points[1]) * 100
            print(f"Distance betwen two point is {d} cm")
            distances.append(d)
            collect_two_points = []


is_first = True
while True:

    imgs = engine.get_image()

    cv2.namedWindow('image')
    cv2.imshow('image', imgs['rgb'])    
    cv2.setMouseCallback('image', onMouse)
    # is_first=False

    if cv2.waitKey(1) == ord('q'):
        ds = np.array(distances) * 10
        print(f"points distances { np.mean(ds)}( {np.std(ds)}) mm")
        break
 
cv2.destroyAllWindows()
engine.close()