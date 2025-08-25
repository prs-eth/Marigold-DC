#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python script/eval.py \
    --data_dir ${BASE_DATA_DIR}/void150 \
    --prediction_dir output/${subfolder}/void150/prediction \
    --output_dir output/${subfolder}/void150/eval_metric