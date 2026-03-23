root_dir="${1:-/home/ben/gasv2_data/}"
evaltag=${2:-""} # first bash argumen, 
seed=${3:-11}  # second bash arguemnt 
online_eps=${4:-100} # 3rd bash argument , "" , "object_scale1", "object_scale2", "object_type1", "regrasp", "dynamic_cam"


run_with_time () {
  echo ">>> Running: $*"
  start=$(date +%s)
  "$@"
  end=$(date +%s)
  echo "<<< Elapsed: $((end - start)) sec"
  echo
}


source bash/init_surrol.sh

# ======== GASv2
python ./run/rl_train.py --reload-dir \
${root_dir}gasv2/2025_04_22-12_19_18@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0/ \
  --reload-envtag  domain_random_enhance dsa_occup2 ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2" --seed ${seed}


run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv2/2025_03_06-22_11_33@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed1/ \
  --reload-envtag  domain_random_enhance dsa_occup2 ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2" --seed ${seed}


run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv2/2026_03_15-15_59_19@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed2/ \
  --reload-envtag  domain_random_enhance dsa_occup2 ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2" --seed ${seed}


#======== GASv1 ======

run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv1/2025_02_25-15_00_30@grasp_any_v2-domain_random_enhance-dsa_occup2-gasv1@dreamerv2-gas-eval_less-high_oracle3@seed0/ \
  --reload-envtag  domain_random_enhance dsa_occup2 gasv1 ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv1" --seed ${seed}


run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv1/2026_02_26-20_00_20@grasp_any_v2-domain_random_enhance-dsa_occup2-gasv1@dreamerv2-gas-eval_less-high_oracle3@seed1 \
  --reload-envtag  domain_random_enhance dsa_occup2 gasv1 ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv1" --seed 1${seed}

run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv1/2026_03_15-15_56_05@grasp_any_v2-domain_random_enhance-dsa_occup2-gasv1@dreamerv2-gas-eval_less-high_oracle3@seed2 \
  --reload-envtag  domain_random_enhance dsa_occup2 gasv1 ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv1" --seed ${seed}


#====== Dreamerv2 ======
run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}dreamerv2/2025_03_05-15_58_46@grasp_any_v2-domain_random_enhance-dsa_occup2-raw_env@dreamerv2-gas-eval_less-high_oracle3@seed0\
  --reload-envtag  domain_random_enhance dsa_occup2 raw_env ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}DreamerV2" --seed ${seed}

run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}dreamerv2/2026_02_26-20_02_06@grasp_any_v2-domain_random_enhance-dsa_occup2-raw_env@dreamerv2-gas-eval_less-high_oracle3@seed1\
  --reload-envtag  domain_random_enhance dsa_occup2 raw_env ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}DreamerV2" --seed ${seed}


#===== GASv2-Clutch
run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv2-clutch/2025_04_26-16_50_38@grasp_any_v2-domain_random_enhance-dsa_occup2-no_clutch@dreamerv2-gas-eval_less-high_oracle3@seed0\
  --reload-envtag  domain_random_enhance dsa_occup2 no_clutch ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2-Clutch" --seed ${seed}

run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv2-clutch/2026_03_09-02_34_52@grasp_any_v2-domain_random_enhance-dsa_occup2-no_clutch@dreamerv2-gas-high_oracle3@seed1\
  --reload-envtag  domain_random_enhance dsa_occup2 no_clutch ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2-Clutch" --seed ${seed}


#====  GASv2-PID
run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv2-pid/2025_02_25-15_02_12@grasp_any_v2-domain_random_enhance-dsa_occup2-no_pid@dreamerv2-gas-eval_less-high_oracle3@seed0\
  --reload-envtag  domain_random_enhance dsa_occup2 no_pid ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2-PID" --seed ${seed}

run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv2-pid/2026_02_26-20_04_25@grasp_any_v2-domain_random_enhance-dsa_occup2-no_pid@dreamerv2-gas-eval_less-high_oracle3@seed1\
  --reload-envtag  domain_random_enhance dsa_occup2 no_pid ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2-PID" --seed ${seed}


#===  GASv2-RawVR

run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv2-pid/2025_02_25-15_02_12@grasp_any_v2-domain_random_enhance-dsa_occup2-no_pid@dreamerv2-gas-eval_less-high_oracle3@seed0\
  --reload-envtag  domain_random_enhance dsa_occup2 no_pid ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2-PID" --seed ${seed}

run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv2-pid/2026_02_26-20_04_25@grasp_any_v2-domain_random_enhance-dsa_occup2-no_pid@dreamerv2-gas-eval_less-high_oracle3@seed1\
  --reload-envtag  domain_random_enhance dsa_occup2 no_pid  ${evaltag} --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2-PID" --seed ${seed}


#==== GASv2-DR
run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gas-nodr/2025_04_22-12_17_55@grasp_any_v2-domain_random_enhance-dsa_occup2-no_dr@dreamerv2-gas-eval_less-high_oracle3@seed0\
  --reload-envtag  domain_random_enhance dsa_occup2 no_dr ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2-DR" --seed ${seed}

run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gas-nodr/2026_03_12-19_36_25@grasp_any_v2-domain_random_enhance-dsa_occup2-no_dr@dreamerv2-gas-high_oracle3@seed2\
  --reload-envtag  domain_random_enhance dsa_occup2 no_dr ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2-DR" --seed ${seed}



# === GASv2-BC
run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv2-bc/2026_02_27-18_23_38@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2_bc-gas@seed0 \
  --reload-envtag  domain_random_enhance dsa_occup2 ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2-BC" --seed ${seed}

run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv2-bc/2026_03_06-16_25_26@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2_bc-gas@seed1 \
  --reload-envtag  domain_random_enhance dsa_occup2 ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2-BC" --seed ${seed}

run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}gasv2-bc/2026_03_10-10_04_17@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2_bc-gas@seed2 \
  --reload-envtag  domain_random_enhance dsa_occup2 ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}GASv2-BC" --seed ${seed}



source bash/init_surrol_ppo.sh
# === PPO 
run_with_time python ./run/rl_train.py --reload-dir \
${root_dir}ppo/2025_05_30-21_53_28@grasp_any_v2-domain_random_enhance-dsa_occup2-raw_env@ppo-high_oracle3@seed0 \
  --reload-envtag  domain_random_enhance raw_env ${evaltag}  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps ${online_eps} --save-prefix "${evaltag:+${evaltag}-}PPO" --seed ${seed}
