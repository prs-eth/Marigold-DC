#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python script/eval.py \
    --data_dir ${BASE_DATA_DIR}/scannet \
    --prediction_dir output/${subfolder}/scannet/prediction \
    --output_dir output/${subfolder}/scannet/eval_metric