# Evaluate (multi-modal)
python -m gpt2_dst.scripts.evaluate_response_BART \
    --input_path_target="./data/simmc2.1_dials_dstc11_devtest_target.txt" \
    --input_path_predicted="./results/simmc2.1_dials_dstc11_devtest_predicted.txt" \
    --output_path_report="./results/simmc2.1_dials_dstc11_devtest_response.txt"