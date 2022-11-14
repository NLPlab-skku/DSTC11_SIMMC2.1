#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    PATH_DATA_DIR=$(realpath ../data)
else
    PATH_DIR=$(realpath "$1")
    PATH_DATA_DIR=$(realpath "$2")
fi

# Train split
python3 -m scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_train.json \
    --output_path_predict="${PATH_DIR}"/data/simmc2.1_dials_dstc11_train_predict.txt \
    --output_path_target="${PATH_DIR}"/data/simmc2.1_dials_dstc11_train_target.txt \
    --len_context=5 \
    --use_multimodal_contexts=1 \
    --output_path_special_tokens="${PATH_DIR}"/data/simmc2.1_special_tokens.json

# Dev split
python3 -m scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_dev.json \
    --output_path_predict="${PATH_DIR}"/data/simmc2.1_dials_dstc11_dev_predict.txt \
    --output_path_target="${PATH_DIR}"/data/simmc2.1_dials_dstc11_dev_target.txt \
    --len_context=5 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/data/simmc2.1_special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/gdata/simmc2.1_special_tokens.json \

# Devtest split
python3 -m scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_devtest.json \
    --output_path_predict="${PATH_DIR}"/data/simmc2.1_dials_dstc11_devtest_predict.txt \
    --output_path_target="${PATH_DIR}"/data/simmc2.1_dials_dstc11_devtest_target.txt \
    --len_context=5 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/data/simmc2.1_special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/data/simmc2.1_special_tokens.json \
    