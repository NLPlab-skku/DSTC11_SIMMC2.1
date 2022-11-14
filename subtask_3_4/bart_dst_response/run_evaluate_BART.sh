#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Evaluate (multi-modal)
python -m scripts.evaluate \
    --input_path_target="${PATH_DIR}"/data/simmc2.1_dials_dstc11_devtest_target.txt \
    --input_path_predicted="${PATH_DIR}"/results/simmc2.1_dials_dstc11_devtest_predicted.txt \
    --output_path_report="${PATH_DIR}"/results/simmc2.1_dials_devtest_report.json