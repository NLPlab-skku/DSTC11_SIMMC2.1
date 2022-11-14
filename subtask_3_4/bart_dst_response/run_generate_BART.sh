#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (multi-modal)
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_generation_BART \
    --model_type=facebook/bart-base \
    --model_name_or_path="${PATH_DIR}"/save/model/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<end>' \
    --prompts_from_file="${PATH_DIR}"/data/simmc2.1_dials_dstc11_devtest_predict.txt \
    --path_output="${PATH_DIR}"/results/simmc2.1_dials_dstc11_devtest_predicted.txt\
