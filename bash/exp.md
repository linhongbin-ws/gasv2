# Prepare

put object in the view of camera
```sh
python ./run/cam_test.py
```

calibrate cam distance to center of workspace
```sh
python ./run/calibrate_cam_dis.py 
```

calibrate workspace limit
```sh
python ./run/calibrate_dvrk_ws.py 
```

label segmentation
```sh
python ./run/label_dvrk.py
```

env play
```sh
python ./run/env_play.py  --env-tag domain_random_enhance dsa_occup2 gasv2_dvrk --oracle keyboard --action oracle
```

# Training
GASV2
```sh
python ./run/rl_train.py  --env-tag domain_random_enhance dsa_occup2  --baseline-tag gas high_oracle3 
```
GASV1
```sh
python ./run/rl_train.py --env-tag domain_random_enhance dsa_occup2  gasv1 --baseline-tag gas eval_less high_oracle3 
```

dreamerv2
```sh
python ./run/rl_train.py --env-tag  domain_random_enhance dsa_occup2  raw_env --baseline-tag  gas eval_less high_oracle3 
```

GASv2-rawRGB
```sh
python ./run/rl_train.py --env-tag  domain_random_enhance dsa_occup2  no_dsa --baseline-tag gas eval_less high_oracle3 
```
GASv2-rawControl
```sh
python ./run/rl_train.py --env-tag  domain_random_enhance dsa_occup2  no_pid --baseline-tag gas eval_less high_oracle3
```

PPO
```sh
python ./run/rl_train.py --env-tag domain_random_enhance dsa_occup2  raw_env --baseline ppo --baseline-tag high_oracle3
```


# eval on surrol

Performance Study
gasv2
```sh
python ./run/rl_train.py --reload-dir ./data/agent/2025_02_13-22_25_34@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0  --reload-envtag  domain_random_enhance dsa_occup2   --online-eval --novis --vis-tag obs rgb dsa mask --online-eps 100 --save-prefix xxx --seed 4
```


# eval on dvrk
```sh
python ./run/dvrk_eval.py --reload-dir ./data/agent/gasv2/2025_03_06-22_11_33@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/1.8m/  --reload-envtag  domain_random_enhance dsa_occup2 gasv2_dvrk --online-eval --visualize --vis-tag obs rgb dsa mask depth --online-eps 20 --save-prefix dVRK-Performance-GASV2 --seed 1
```

```sh
python ./run/dvrk_eval.py --reload-dir ./log/2025_01_02-14_02_12@grasp_any_v2-action_continuous@dreamerv2-gasv2@seed0/  --reload-envtag  gasv2_dvrk action_continuous --online-eval --visualize --vis-tag obs rgb dsa mask --online-eps 20 --save-prefix xxx


python ./run/dvrk_eval.py --reload-dir ./data/agent/2025_02_13-22_25_34@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0  --reload-envtag  domain_random_enhance dsa_occup2 gasv2_dvrk --online-eval --visualize --vis-tag obs rgb dsa mask --online-eps 20 --save-prefix xxx

./data/agent/2025_03_06-22_11_33@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/1.8m/