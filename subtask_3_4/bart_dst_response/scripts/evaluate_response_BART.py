#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

    Scripts for evaluating the GPT-2 DST model predictions.

    First, we parse the line-by-line stringified format into responses
    and compute BLEU score.
"""
import argparse
import json
from utils.convert_BART import parse_flattened_results_from_file
from utils.evaluate_dst import evaluate_from_flat_list

import nltk
import numpy as np

CLS_TOKEN = '<cls>'
SEP_1_TOKEN = '<sep1>'
SEP_2_TOKEN = '<sep2>'
END_TOKEN = '<end>'

# last_uttr = splits[1].strip().replace(END_TOKEN, "").strip()


def normalize_sentence(sentence):
    """Normalize the sentences and tokenize."""
    return nltk.tokenize.word_tokenize(sentence.lower())


def parse_response_from_file(input_path):
    """Parses the response from a flattened file.

    Args:
        input_path: Path to read the responses from.
    """
    lines = []
    with open(input_path, "r") as file_id:
        for ii in file_id.readlines():
#             print("II : {}".format(ii))
            split_line = ii.split(SEP_2_TOKEN)
            if len(split_line) == 1:       # split_line 길이가 1이라는 건, EOB 토큰이 없다는 거고 생성 실패했다는 소리
                print("split_line : {}".format(split_line))            
            
                last_uttr = split_line[0].strip("\n").replace("<cls>","").replace("<pad>","").replace("<end>","").lstrip().rstrip()
                print("last_uttr : {}".format(last_uttr))            

                lines.append(("", last_uttr))
            else:
                prompt = split_line[0].strip("\n").strip(" ")
                last_uttr = split_line[1].strip("\n").replace("<cls>","").replace("<pad>","").replace("<end>","").lstrip().rstrip()
                lines.append(
                    (prompt, last_uttr)
                )
#         a = split_line[0].strip("\n")
#         print("\n\nsplit_line[0] : {}".format(a))

#         a = split_line[1].strip("\n").strip("<EOS>").lstrip()
#         print("\nsplit_line[1] : {}".format(a))
        
    return lines


if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path_target", help="path for target, line-separated format (.txt)"
    )
    parser.add_argument(
        "--input_path_predicted",
        help="path for model prediction output, line-separated format (.txt)",
    )
    parser.add_argument(
        "--output_path_report", help="path for saving evaluation summary (.json)"
    )

    args = parser.parse_args()
    input_path_target = args.input_path_target
    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_report

    # Convert the data from the GPT-2 friendly format to JSON
    list_target = parse_response_from_file(input_path_target)
    list_predicted = parse_response_from_file(input_path_predicted)

    # Compute BLEU scores.
    bleu_scores = []
    # Smoothing function.
    chencherry = nltk.translate.bleu_score.SmoothingFunction()

    for response, gt_response in zip(list_predicted, list_target):
#         print("\nPred : {}".format(len(list(response[0]))))
#         print("Generate : {}".format(len(list(gt_response[0]))))

#         pp = list()
#         gg = list()
#         for p, g in zip(list(response[0]), list(gt_response[0])):
#             if not p==g:
#                 pp.append(p)
#                 gg.append(g)
#         print("\nPP : {}".format(("").join(pp)))
#         print("GG : {}".format(("").join(gg)))
                    
#         assert response[0] == gt_response[0], "Input contexts do not match!"
        bleu_score = nltk.translate.bleu_score.sentence_bleu(
            [normalize_sentence(gt_response[1])],
            normalize_sentence(response[1]),
            smoothing_function=chencherry.method7,
        )
        bleu_scores.append(bleu_score)
    print(
        "BLEU score: {} +- {}".format(
            np.mean(bleu_scores), np.std(bleu_scores) / np.sqrt(len(bleu_scores))
        )
    )
