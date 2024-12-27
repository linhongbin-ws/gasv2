in_dir=${1}
dir="./data/logzip/"
mkdir -p ${dir}
filename=`echo $dir$(basename $in_dir)`
zip -r ${filename}.zip ${in_dir} -i "*variables.pkl" "*events*" "*.yaml" "*.jsonl" -j