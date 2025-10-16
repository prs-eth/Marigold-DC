#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python script/eval.py \
    --data_dir ${BASE_DATA_DIR}/kittidc \
    --prediction_dir output/${subfolder}/kittidc/prediction \
    --output_dir output/${subfolder}/kittidc/eval_metric