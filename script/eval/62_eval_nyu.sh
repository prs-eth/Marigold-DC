#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python script/eval.py \
    --data_dir ${BASE_DATA_DIR}/nyu \
    --prediction_dir output/${subfolder}/nyu/prediction \
    --output_dir output/${subfolder}/nyu/eval_metric