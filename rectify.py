import cv2 as cv
import numpy as np

def my_rectify(left_image, right_image, fs):

    fn_R = fs.getNode("R")
    fn_T = fs.getNode("T")
    fn_M1= fs.getNode("M1")
    fn_D1 = fs.getNode("D1")
    fn_M2= fs.getNode("M2")
    fn_D2 = fs.getNode("D2")

    stereo_R = fn_R.mat().astype(np.float64)
    stereo_T = fn_T.mat().astype(np.float64)
    # stereo_T = fn_T.mat().astype(np.float64).transpose()

    stereo_M1 = fn_M1.mat()
    stereo_D1 = fn_D1.mat()
    stereo_M2 = fn_M2.mat()
    stereo_D2 = fn_D2.mat()

    # print(stereo_R)
    # print(stereo_T.shape)
    # print(stereo_M1)
    # print(stereo_D1)
    # print(stereo_M2)
    # print(stereo_D2)
    height, width, channel = left_image.shape

    # left_image_undistorted = cv.undistort(left_image, stereo_M1, stereo_D1)
    # right_image_undistorted = cv.undistort(right_image, stereo_M2, stereo_D2)

    R1, R2, P1, P2, Q, roi_left, roi_right = cv.stereoRectify(stereo_M1, stereo_D1, stereo_M2, stereo_D2, (width, height), stereo_R, stereo_T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0)

    #print(P1)
    #print(P2)
    #print(Q)
    #exit()

    leftMapX, leftMapY = cv.initUndistortRectifyMap(stereo_M1, stereo_D1, R1, P1, (width, height), cv.CV_32FC1)
    left_rectified = cv.remap(left_image, leftMapX, leftMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

    rightMapX, rightMapY = cv.initUndistortRectifyMap(stereo_M2, stereo_D2, R2, P2, (width, height), cv.CV_32FC1)
    right_rectified = cv.remap(right_image, rightMapX, rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

    # print(left_rectified.shape)
    # print(right_rectified.shape)

    return left_rectified, right_rectified

def get_F(fs):

    fn_R = fs.getNode("R")
    fn_T = fs.getNode("T")
    fn_M1= fs.getNode("M1")
    fn_D1 = fs.getNode("D1")
    fn_M2= fs.getNode("M2")
    fn_D2 = fs.getNode("D2")

    stereo_R = fn_R.mat().astype(np.float64)
    stereo_T = fn_T.mat().astype(np.float64).transpose()

    stereo_M1 = fn_M1.mat().astype(np.float64)
    stereo_D1 = fn_D1.mat()
    stereo_M2 = fn_M2.mat().astype(np.float64)
    stereo_D2 = fn_D2.mat()
    # print("M1", stereo_M1)
    # print("M2", stereo_M2)

    K1_inv_t = np.linalg.inv(stereo_M1.transpose())
    K2_inv = np.linalg.inv(stereo_M2)

    T_mat = np.array(
            [[0.0,-stereo_T[2][0],stereo_T[1][0]],
            [stereo_T[2][0],0.0,-stereo_T[0][0]],
            [-stereo_T[1][0],stereo_T[0][0],0.0]])

    F = np.dot(K1_inv_t, np.dot(np.dot(T_mat, stereo_R), K2_inv))

    return F




if __name__ == '__main__':

    left_image_path = "./ext/python_stereo_camera_calibrate/frames_pair/camera0_0.png"

    right_image_path = "./ext/python_stereo_camera_calibrate/frames_pair/camera1_0.png"

    left_image = cv.imread(left_image_path)
    right_image = cv.imread(right_image_path)

    fs = cv.FileStorage("./ext/python_stereo_camera_calibrate/cam_cal.yaml", cv.FILE_STORAGE_READ)

    left_rectified, right_rectified = my_rectify(left_image, right_image, fs)

    cv.imwrite("./left_image.jpg",left_rectified)
    cv.imwrite("./right_image.jpg",right_rectified)

    cv.waitKey(0)
