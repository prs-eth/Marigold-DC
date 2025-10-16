#!/usr/bin/env bash
set -e
set -x

# Use specified output path, otherwise, default value
subfolder=${1:-"eval"}

python script/infer.py \
    --input_dir ${BASE_DATA_DIR}/void1500 \
    --output_dir output/${subfolder}/void1500 \
    --num_inference_steps 50 \
    --ensemble_size 10 \
    --processing_resolution 0 \
    --seed 2024