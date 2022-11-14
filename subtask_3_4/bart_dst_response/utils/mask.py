import os
import numpy as np
import torch
import torch.nn as nn
import random
import math
from tqdm import tqdm
from collections import defaultdict
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch.nn.utils.rnn import pad_sequence

class Masker(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer    
        
    def compute_masked_indices(self, inputs, model, mlm_probability):
        raise NotImplementedError
    
    def gen_inputs_labels(self, inputs, masked_indices):
        raise NotImplementedError
        
    def mask_tokens(self, inputs, mlm_probability = 0.15):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        masked_indices = self.compute_masked_indices(inputs, mlm_probability)
        return self.gen_inputs_labels(inputs, masked_indices)

class BertMasker(Masker):
    def compute_masked_indices(self, inputs, mlm_probability):
        # inputs : (batch, max_seq_length)
        # probability_matrix : (batch, max_seq_length)
        # probability_matrix[0] : [0.1500, 0.1500, ..., 0.1500]
        probability_matrix = torch.full(inputs.shape, mlm_probability)        

        # special_tokens_mask : batch (length)
        # special_tokens_mask[0] : max_seq_length (length)
        # special_tokens_mask[0] : [1, 0, 0, 0, ..., 0, 1, 1, 1, ..., 1]
        # CLS, SEP, PAD 토큰은 Masking하지 않기 위해 1로 표시하는 듯 하다        

        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()]
        # pos_token = 50265
        # neg_token = 50266
        # for idx, tokens in enumerate(inputs.detach().numpy().tolist()):
        #     if pos_token in tokens or neg_token in tokens:
        #         print("\ninputs : {}".format(inputs[idx]))
        #         print("special_tokens_mask : {}".format(special_tokens_mask[idx]))

        # probability_matrix : (batch, max_seq_length)
        # probability_matrix[0] : [0.0000, 0.1500, ..., 0.0000]
        # Special Token에 해당하는 위치는 확률을 0으로
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)    # torch.bool 은 오류가 발생 -> uint8 사용 (최신 pytorch 버전에서는 bool이 사라진 듯)

        # masked_indices : (batch, max_seq_length)
        # masked_indices[0] : tensor([False, True, False, False, True, ..., False, False])
        # 15% 확률로 bool type 변경 -> Masking할 단어 선정
        masked_indices = torch.bernoulli(probability_matrix).type(torch.bool)
        return masked_indices

    def gen_inputs_labels(self, inputs, masked_indices):
        # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
        # labels : (batch, max_seq_length)        
        # labels[0] : tensor([-100, -100, ..., 1056, -100, 234, -100, ..., -100])
        inputs = inputs.clone()
        labels = inputs.clone()

        # labels[0] : tensor([-100, -100, ..., 1056, -100, 234, -100, ..., -100])
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token])[0]

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.bool) & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.vocab), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

