source bash/init_surrol.sh

# python ./run/rl_train.py  --env-tag domain_random_enhance dsa_occup2  --baseline-tag gas --baseline dreamerv2_bc --seed 3

# python ./run/rl_train.py  --env-tag domain_random_enhance dsa_occup2  --baseline-tag gas --baseline dreamerv2_bc --seed 4

python ./run/rl_train.py --env-tag domain_random_enhance dsa_occup2  gasv1 --baseline-tag gas eval_less high_oracle3 --seed 2

python ./run/rl_train.py --env-tag domain_random_enhance dsa_occup2  gasv1 --baseline-tag gas eval_less high_oracle3 --seed 4

python ./run/rl_train.py  --env-tag domain_random_enhance dsa_occup2 no_clutch  --baseline-tag gas eval_less high_oracle3   --seed 2

python ./run/rl_train.py  --env-tag domain_random_enhance dsa_occup2 no_clutch  --baseline-tag gas eval_less high_oracle3   --seed 3

python ./run/rl_train.py  --env-tag domain_random_enhance dsa_occup2 no_clutch  --baseline-tag gas eval_less high_oracle3   --seed 4