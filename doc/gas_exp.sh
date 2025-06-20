source bash/init_dvrk.sh
evaltag=${1:-""} # second bash argument 
seed=${2:-1}  # first bash arguemnt 
online_eps=${3:-10} # second bash argument 



# # dvrk performance
# GASv2
# python ./run/dvrk_eval.py \
#  --online-eval --vis-tag obs rgb dsa mask depth --online-eps ${online_eps}  \
#  --save-prefix dvrk-${evaltag}-GASv2 --seed ${seed} \
#  --reload-dir ./data/agent/gasv2/2025_03_06-22_11_33@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/achive/1.8m/  \
#  --reload-envtag domain_random_enhance dsa_occup2 gasv2_dvrk \
#  --visualize 



# # - GASv2-NoCluth
# python ./run/dvrk_eval.py \
#  --online-eval --vis-tag obs rgb dsa mask depth --online-eps ${online_eps}  \
#  --save-prefix  dvrk-${evaltag}-GASv2-NoCluth --seed ${seed} \
#  --reload-dir ./data/agent/gasv2-noclutch/2025_04_26-16_50_38@grasp_any_v2-domain_random_enhance-dsa_occup2-no_clutch@dreamerv2-gas-eval_less-high_oracle3@seed0/ \
#  --reload-envtag domain_random_enhance dsa_occup2 no_clutch gasv2_dvrk \
#  --visualize 

# # GASv1
# python ./run/dvrk_eval.py \
#  --online-eval --vis-tag obs rgb dsa mask depth --online-eps ${online_eps}  \
#  --save-prefix dvrk-${evaltag}-GASv1 --seed ${seed} \
#  --reload-dir ./data/agent/gasv1/2025_02_25-15_00_30@grasp_any_v2-domain_random_enhance-dsa_occup2-gasv1@dreamerv2-gas-eval_less-high_oracle3@seed0/ \
#  --reload-envtag domain_random_enhance dsa_occup2 gasv1 gasv2_dvrk \
#  --visualize 

# # dreamerv2
# python ./run/dvrk_eval.py \
#  --online-eval --vis-tag obs rgb dsa mask depth --online-eps ${online_eps}  \
#  --save-prefix dvrk-${evaltag}-Dreamerv2 --seed ${seed}\
#  --reload-dir ./data/agent/dreamerv2/2025_03_05-15_58_46@grasp_any_v2-domain_random_enhance-dsa_occup2-raw_env@dreamerv2-gas-eval_less-high_oracle3@seed0/ \
#  --reload-envtag domain_random_enhance dsa_occup2 raw_env gasv2_dvrk \
#  --visualize 

# # GASv2_rawControl
# python ./run/dvrk_eval.py \
#  --online-eval --vis-tag obs rgb dsa mask depth --online-eps ${online_eps} \
#  --save-prefix dvrk-${evaltag}-GASv2_rawControl --seed ${seed}\
#  --reload-dir ./data/agent/gasv2-raw_control/2025_02_25-15_02_12@grasp_any_v2-domain_random_enhance-dsa_occup2-no_pid@dreamerv2-gas-eval_less-high_oracle3@seed0 \
#  --reload-envtag domain_random_enhance dsa_occup2 no_pid gasv2_dvrk \
#  --visualize 


# # GASv2_rawRGB
# python ./run/dvrk_eval.py \
#  --online-eval --vis-tag obs rgb dsa mask depth --online-eps ${online_eps}  \
#  --save-prefix dvrk-${evaltag}-GASv2_rawRGB --seed ${seed}\
#  --reload-dir data/agent/gasv2-raw_rgb/2025_02_25-15_01_30@grasp_any_v2-domain_random_enhance-dsa_occup2-no_dsa@dreamerv2-gas-eval_less-high_oracle3@seed0 \
#  --reload-envtag domain_random_enhance dsa_occup2 no_dsa gasv2_dvrk \
#  --visualize 



# # GASv2_NoDR
# python ./run/dvrk_eval.py \
#  --online-eval --vis-tag obs rgb dsa mask depth --online-eps ${online_eps}  \
#  --save-prefix dvrk-${evaltag}-GASv2_nodr --seed ${seed}\
#  --reload-dir ./data/agent/gas-nodr/2025_04_22-12_17_55@grasp_any_v2-domain_random_enhance-dsa_occup2-no_dr@dreamerv2-gas-eval_less-high_oracle3@seed0 \
#  --reload-envtag domain_random_enhance dsa_occup2 no_dr gasv2_dvrk \
#  --visualize 


 
















 # # - GASv2
# python ./run/dvrk_eval.py \
#  --online-eval --vis-tag obs rgb dsa mask depth --online-eps ${online_eps}  \
#  --save-prefix dvrk-${evaltag}-GASv2 --seed ${seed} \
#  --reload-dir ./data/agent/gasv2/2025_04_22-12_19_18@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/  \
#  --reload-envtag domain_random_enhance dsa_occup2 gasv2_dvrk \
#  --visualize 


############### robustness study
# #action noise

# # GASv2
# python ./run/dvrk_eval.py \
#  --online-eval --vis-tag obs rgb dsa mask depth --online-eps ${online_eps}  \
#  --save-prefix dvrk-${evaltag}-GASv2 --seed ${seed} \
#  --reload-dir ./data/agent/gasv2/2025_03_06-22_11_33@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/achive/1.8m/  \
#  --reload-envtag domain_random_enhance dsa_occup2 gasv2_dvrk action_noise \
#  --visualize 


 # # image noise
 python ./run/dvrk_eval.py \
 --online-eval --vis-tag obs rgb dsa mask depth --online-eps ${online_eps}  \
 --save-prefix dvrk-${evaltag}-GASv2 --seed ${seed} \
 --reload-dir ./data/agent/gasv2/2025_03_06-22_11_33@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/achive/1.8m/  \
 --reload-envtag domain_random_enhance dsa_occup2 gasv2_dvrk image_noise \
 --visualize 