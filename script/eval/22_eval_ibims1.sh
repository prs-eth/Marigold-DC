#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python script/eval.py \
    --data_dir ${BASE_DATA_DIR}/ibims1 \
    --prediction_dir output/${subfolder}/ibims1/prediction \
    --output_dir output/${subfolder}/ibims1/eval_metric