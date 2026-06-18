log_dir="${1:-./log}"
out_dir="./data/log_zip_all"
mkdir -p ${out_dir}

for exp in ${log_dir}/*/; do
    name=$(basename ${exp})
    out="${out_dir}/${name}.zip"
    if [ -f "${out}" ]; then
        echo "skip: ${name}"
        continue
    fi
    echo "zipping: ${name}"
    zip -r ${out} ${exp} -i "*variables.pkl" "*events*" "*.yaml" "*.jsonl" -j
done
echo "done. output: ${out_dir}"
