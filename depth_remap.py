import numpy as np
import cv2
def depth_remap(image_in, in_dis, out_dis):
    alpha = in_dis / out_dis
    w = int(image_in.shape[0] * alpha) 
    h = int(image_in.shape[1] * alpha)
    if isinstance(image_in[0][0], np.bool_):
        image = np.zeros(image_in.shape, dtype=np.uint8)
        image[image_in] = 255
    else:
        image = image_in
    pix = cv2.resize(image, (w, h))
    if isinstance(image_in[0][0], np.bool_):
        new_image = np.zeros(image.shape, dtype=bool)
        pix = pix == 255
    else: 
        new_image = np.zeros(image.shape, dtype=np.uint8)
    cx1 = image.shape[0] // 2 - w // 2
    cx2 = cx1 + pix.shape[0]
    cy1 = image.shape[1] // 2 - h // 2
    cy2 = cy1 + pix.shape[1]
    if len(new_image.shape) == 3:
        new_image[cx1:cx2, cy1:cy2, :] = pix
    elif len(new_image.shape) == 2:
        new_image[cx1:cx2, cy1:cy2] = pix
    return new_image

def RGBDTransform(rgb, depth_real, fx, fy, cx, cy, Ts=[]):
    assert len(Ts) != 0
    height, width, channel = rgb.shape[0], rgb.shape[1], rgb.shape[2]
    zs = depth_real.copy().reshape(-1)
    rs = rgb[:, :, 0].copy().reshape(-1)
    gs = rgb[:, :, 1].copy().reshape(-1)
    bs = rgb[:, :, 2].copy().reshape(-1)
    rgb_u = np.stack(width * [np.arange(width)], axis=1)
    rgb_v = np.stack(height * [np.arange(height)], axis=0)
    rgb_u_arr = rgb_u.reshape(-1)
    rgb_v_arr = rgb_v.reshape(-1)
    xs = np.multiply((rgb_u_arr - cy) / (fx), zs)
    ys = np.multiply((rgb_v_arr - cx) / (fy), zs)

    new_rgbs = []
    for T in Ts:
        assert T.shape[0] == 4 and T.shape[1] == 4
        xyz = np.stack([xs, ys, zs, np.ones(xs.shape)], axis=0)
        print(T.shape, xyz.shape)
        new_xyz = np.matmul(T, xyz)
        new_x = new_xyz[0, :]
        new_y = new_xyz[1, :]
        new_z = new_xyz[2, :]
        print(fx * np.divide(new_x, new_z) + cy)
        new_us = fx * np.divide(new_x, new_z) + cy
        new_vs = fy * np.divide(new_y, new_z) + cx
        new_us = np.around(new_us).astype(np.int)
        new_vs = np.around(new_vs).astype(np.int)

        new_us = new_us.reshape(-1)
        new_vs = new_vs.reshape(-1)
        new_rgb = np.zeros(
            (
                width,
                height,
                channel,
            ),
            dtype=np.uint8,
        )
        new_depth = -np.ones(
            (
                width,
                height,
            ),
            dtype=np.float,
        )
        for i in range(new_us.shape[0]):
            u = new_us[i]
            v = new_vs[i]
            if (u < 0) or (u > width - 1):
                continue
            if (v < 0) or (v > height - 1):
                continue
            if new_depth[u, v] >= zs[i]:
                continue

            new_rgb[u, v, 0] = rs[i]
            new_rgb[u, v, 1] = gs[i]
            new_rgb[u, v, 2] = bs[i]
        new_rgbs.append(new_rgb)
    return new_rgbs


if __name__ == "__main__":
    import cv2
    from matplotlib.pyplot import imshow, subplot, axis, cm, show

    rgb = cv2.imread("./test_record/frame1_1.png")
    depth_px = cv2.imread("./test_record/pred_depth_1.png")

    rgb = cv2.resize(rgb, (600,600))
    depth_px = cv2.resize(depth_px, (600, 600))
    print(rgb.shape, depth_px.shape)

    width = 600
    height = 600
    fov = 42 / 180 * np.pi
    fx = width / 2 / np.tan(fov / 2)
    fy = fx
    cx = (width - 1) / 2
    cy = (height - 1) / 2

    from gym_ras.tool.common import scale_arr, getT

    depth_real = scale_arr(depth_px[:, :, 0], 0, 255, 0.05, 0.25)

    T0 = getT([0,0,0.1,], [0,0,0], rot_type="euler", euler_convension="xyz", euler_Degrees=True)
    Ts = [T0]
    new_rgbs = RGBDTransform(rgb,depth_real,fx,fy, cx,cy, Ts)

    subplot(1, 3, 1)
    axis("off")
    imshow(rgb)

    subplot(1, 3, 2)
    axis("off")
    imshow(new_rgbs[0])

    remap_rgb = depth_remap(rgb, 0.15, 0.3)
    subplot(1, 3, 3)
    axis("off")
    imshow(remap_rgb)
    show()
