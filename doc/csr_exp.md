python ./run/dvrk_eval.py --reload-dir ./data/agent/2024_01_21-13_57_13@ras-gas_surrol@dreamerv2-gas@seed1 --reload-envtag gas_surrol csr_grasp_any  --online-eval --visualize --vis-tag obs rgb depth mask --online-eps 20 --save-prefix xxx


python ./run/get_depth_center.py  

python ./run/label2.py


python ./run/move_gripper_to_correct_ori.py

python run/stream_cam_csr.py --vis-tag rgb depth

python ./run/calibrate_csr_ws.py

note:

cam configruation: 45%, 20cm, image center locates the center of workspace

label: psm except gripper has to be consistent with RSS paper


