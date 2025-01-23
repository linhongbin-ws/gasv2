# Training
GASV2
```sh
python ./run/rl_train.py --baseline-tag gas no_oracle  --env-tag domain_random_enhance dsa_occup2  
```
GASV1
```sh
python ./run/rl_train.py --env-tag gasv1 --baseline-tag gas eval_less
```

dreamerv2
```sh
python ./run/rl_train.py --env-tag raw_env --baseline-tag gas eval_less
```

GASv2-rawRGB
```sh
python ./run/rl_train.py --env-tag no_dsa --baseline-tag gas eval_less
```
GASv2-rawControl
```sh
python ./run/rl_train.py --env-tag no_pid --baseline-tag gas eval_less
```


# eval on dvrk
python ./run/dvrk_eval.py --reload-dir ./data/agent/2024_12_30-10_58_55@grasp_any_v2@dreamerv2-gasv2@seed0  --reload-envtag  gasv2_dvrk --online-eval --visualize --vis-tag obs rgb dsa mask --online-eps 20 --save-prefix xxx


python ./run/dvrk_eval.py --reload-dir ./log/2025_01_02-14_02_12@grasp_any_v2-action_continuous@dreamerv2-gasv2@seed0/  --reload-envtag  gasv2_dvrk action_continuous --online-eval --visualize --vis-tag obs rgb dsa mask --online-eps 20 --save-prefix xxx