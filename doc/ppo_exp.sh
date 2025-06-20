source bash/init_dvrk_ppo.sh
evaltag=${1:-""} # second bash argument 
seed=${2:-1}  # first bash arguemnt 
online_eps=${3:-10} # second bash argument 



# GASv2_NoDR
python ./run/dvrk_eval.py \
 --online-eval --vis-tag obs rgb dsa mask depth --online-eps ${online_eps}  \
 --save-prefix dvrk-${evaltag}-ppo --seed ${seed}\
 --reload-dir ./data/agent/ppo/2025_05_30-21_53_28@grasp_any_v2-domain_random_enhance-dsa_occup2-raw_env@ppo-high_oracle3@seed0 \
 --reload-envtag domain_random_enhance dsa_occup2 raw_env gasv2_dvrk \
 --visualize 


 