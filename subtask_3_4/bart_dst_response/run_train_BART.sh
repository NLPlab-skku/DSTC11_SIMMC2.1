#!/bin/bash
# Train (multi-modal)
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_language_modeling_BART \
    --output_dir="./save/model" \
    --model_type=facebook/bart-base \
    --model_name_or_path=facebook/bart-base \
    --do_train \
    --add_special_tokens="./data/simmc2.1_special_tokens.json" \
    --train_data_pred_file="./data/simmc2.1_dials_dstc11_train_predict.txt" \
    --train_data_target_file="./data/simmc2.1_dials_dstc11_train_target.txt" \
    --do_eval \
    --seed=12345 \
    --eval_data_pred_file="./data/simmc2.1_dials_dstc11_dev_predict.txt" \
    --eval_data_target_file="./data/simmc2.1_dials_dstc11_dev_target.txt" \
    --num_train_epochs=10 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=2 \
    --per_gpu_eval_batch_size=2 \
    --save_steps=15000 \
    --logging_steps=500 \
    --gradient_accumulation_steps=2
