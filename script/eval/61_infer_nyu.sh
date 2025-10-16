#!/usr/bin/env bash
set -e
set -x

# Use specified output path, otherwise, default value
subfolder=${1:-"eval"}

python script/infer.py \
    --input_dir ${BASE_DATA_DIR}/nyudepthv2 \
    --output_dir output/${subfolder}/nyudepthv2 \
    --num_inference_steps 50 \
    --ensemble_size 10 \
    --processing_resolution 768 \
    --seed 2024