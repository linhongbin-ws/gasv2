seed=${1:-1}  # first bash arguemnt 
online_eps=${2:-100} # second bash argument 
source bash/init_surrol.sh






# performance
GASv2
python ./run/rl_train.py  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix surrol-performance-GASv2 --seed ${seed} \
 --reload-dir ./data/agent/gasv2/2025_03_06-22_11_33@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/achive/1.8m/

# GASv1
python ./run/rl_train.py  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix surrol-performance-GASv1 --seed ${seed} \
 --reload-dir ./data/agent/gasv1/2025_02_25-15_00_30@grasp_any_v2-domain_random_enhance-dsa_occup2-gasv1@dreamerv2-gas-eval_less-high_oracle3@seed0/

# dreamerv2
python ./run/rl_train.py  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix surrol-performance-Dreamerv2 --seed ${seed}\
 --reload-dir ./data/agent/dreamerv2/2025_03_05-15_58_46@grasp_any_v2-domain_random_enhance-dsa_occup2-raw_env@dreamerv2-gas-eval_less-high_oracle3@seed0/

# GASv2_rawControl
python ./run/rl_train.py  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix surrol-performance-GASv2_rawControl --seed ${seed}\
 --reload-dir ./data/agent/gasv2-raw_control/2025_02_25-15_02_12@grasp_any_v2-domain_random_enhance-dsa_occup2-no_pid@dreamerv2-gas-eval_less-high_oracle3@seed0

# GASv2_rawRGB
python ./run/rl_train.py  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix surrol-performance-GASv2_rawRGB --seed ${seed}\
 --reload-dir data/agent/gasv2-raw_rgb/2025_02_25-15_01_30@grasp_any_v2-domain_random_enhance-dsa_occup2-no_dsa@dreamerv2-gas-eval_less-high_oracle3@seed0

# # PPO
# python ./run/rl_train.py  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix surrol-performance-PPO --seed ${seed}\
#  --reload-dir data/agent/ppo/2025_03_02-18_04_18@grasp_any_v2-domain_random_enhance-dsa_occup2-raw_env@ppo-high_oracle3@seed0


# python ./run/rl_train.py --reload-env-t  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps 100 --save-prefix surrol-performance-GASv2 --seed 3 \
#  --reload-dir ./data/agent/gasv2/2025_03_06-22_11_33@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/achive/1.8m/
#  python ./run/rl_train.py  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps 100 --save-prefix surrol-performance-GASv2 --seed 4 \
#  --reload-dir ./data/agent/gasv2/2025_03_06-22_11_33@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/achive/1.8m/
#  python ./run/rl_train.py  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps 100 --save-prefix surrol-performance-GASv2 --seed 5 \
#  --reload-dir ./data/agent/gasv2/2025_03_06-22_11_33@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/achive/1.8m/
#  python ./run/rl_train.py  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps 100 --save-prefix surrol-performance-GASv2 --seed 6 \
#  --reload-dir ./data/agent/gasv2/2025_03_06-22_11_33@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/achive/1.8m/