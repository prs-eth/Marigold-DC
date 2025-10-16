#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python script/eval.py \
    --data_dir ${BASE_DATA_DIR}/ddad \
    --prediction_dir output/${subfolder}/ddad/prediction \
    --output_dir output/${subfolder}/ddad/eval_metric