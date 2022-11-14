#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)

Adapted from
https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py
"""

import argparse
import logging
import os
import numpy as np
import random
import torch
from transformers import BartForConditionalGeneration as BartLMHeadModel
from transformers import T5ForConditionalGeneration as T5LMHeadModel
from model.bart_mlm import BartForMaskedLM
from utils.arg_parser import generator_parser

from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

args.device = torch.device("cuda")
print(f'There are {torch.cuda.device_count()} GPU(s) available.')
print('Device name:', torch.cuda.get_device_name(0))


#
# Functions to prepare models' input
#

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():

    parser = generator_parser()
    args = parser.parse_args()

    args.device = torch.device("cuda")


    if args.prompts_from_file and not os.path.exists(args.prompts_from_file):
        raise Exception(f"prompt file '{args.prompts_from_file}' not found")

#     # Initialize the model and tokenizer
#     try:
#         args.model_type = args.model_type.lower()
#         model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
#     except KeyError:
#         raise KeyError(
#             "the model {} you specified is not supported. You are welcome to add it and open a PR :)"
#         )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)    
    model = BartForMaskedLM.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    args.length = adjust_length_to_model(
        args.length, max_sequence_length=model.config.max_position_embeddings
    )
    logger.info(args)

    results = []
    prompts = []
    if args.prompts_from_file:
        with open(args.prompts_from_file) as handle:
            prompts = handle.readlines()

    while True:
        if not prompts:
            prompts = [args.prompt if args.prompt else input("Model prompt >>> ")]
            if not args.prompt and (
                len(prompts) == 0
                or prompts[0].strip() == ""
                or prompts[0].lower() == "quit"
            ):
                break  # break while True loop

        n_prompts = len(prompts)
        for i, prompt_text in enumerate(prompts):           
            # Strip any trailing \n if provided
            prompt_text = prompt_text.strip("\n")

            # Different models need different input formatting and/or extra arguments
            requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
            if requires_preprocessing:
                prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
                preprocessed_prompt_text = prepare_input(
                    args, model, tokenizer, prompt_text
                )
                encoded_prompt = tokenizer.encode(
                    preprocessed_prompt_text,
                    add_special_tokens=True,
                    return_tensors="pt",
                    add_space_before_punct_symbol=True,
                )
            else:
                # print("ElSE 문")
                # encoded_prompt = tokenizer.encode(
                #     prompt_text, add_special_tokens=True, return_tensors="pt"
                # )
                ## 최대길이가 넘어가는게 있어서.
                encoded_prompt = tokenizer(
                    [prompt_text], add_special_tokens=True, max_length=1024,truncation=True)
            src = torch.tensor(encoded_prompt.input_ids).to(args.device)
            src_mask = torch.tensor(encoded_prompt.attention_mask).to(args.device)
            # Decode text
            text = tokenizer.decode(
                generated_sequence, clean_up_tokenization_spaces=True
            )
            # print("\nSRC : {} {}".format(src.size(), src))
            try:
                output_sequences = model.generate(
                    src,
                    max_length=args.length + len(src),
                    decoder_start_token_id=tokenizer.pad_token_id,
                    attention_mask=src_mask,
                    early_stopping=True)
            except Exception as e:
                print("\n\nError : {}".format(e))
                print("src type : {}".format(type(src)))
                print("src dim : {}".format(src.size()))
                print("src : {}".format(src[0]))

                print("src_mask type : {}".format(type(src_mask)))
                print("src_mask dim : {}".format(src_mask.size()))
                print("src_mask : {}".format(src_mask[0]))

                print("max_length : {}".format(args.length + len(src)))

                import sys
                sys.exit()
            # print("output_sequences : {} {}".format(output_sequences.size(), output_sequences))

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            generated_sequences = []

            for generated_sequence_idx, generated_sequence in enumerate(
                output_sequences
            ):
                print(
                    "=== GENERATED SEQUENCE {sequence_idx}, {promt_idx}/{n_prompts} ===".format(
                        sequence_idx=generated_sequence_idx + 1,
                        promt_idx=i + 1,
                        n_prompts=n_prompts,
                    )
                )
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(
                    generated_sequence, clean_up_tokenization_spaces=True
                )
                # print("1 TEXT : {}".format(text))
                # print("stop_token : {}".format(args.stop_token))
                # print("SRC : {}".format(src))
                # Remove all text after the stop token
                text = text[: text.find(args.stop_token) if args.stop_token else None]
                text =text.replace("<pad>", "").replace("<s>", "")
                # print("2 TEXT : {}".format(text))
                generated_sequences.append(text)
                
            results.append(generated_sequences)
        prompts = []
        if args.prompt or args.prompts_from_file:
            break  # break while True loop

    if args.path_output is not None:

        # Create a directory if it does not exist
        directory = os.path.dirname(args.path_output)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Format results into a line-separated string file
        str_results = "\n".join(
            [" || ".join(generated_sequences) for generated_sequences in results]
        )
        # Save to a file
        with open(args.path_output, "w") as f_out:
            f_out.write(str_results)

    return results


if __name__ == "__main__":
    main()
