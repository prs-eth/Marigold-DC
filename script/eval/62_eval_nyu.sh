#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python script/eval.py \
    --data_dir ${BASE_DATA_DIR}/nyudepthv2 \
    --prediction_dir output/${subfolder}/nyudepthv2/prediction \
    --output_dir output/${subfolder}/nyudepthv2/eval_metric