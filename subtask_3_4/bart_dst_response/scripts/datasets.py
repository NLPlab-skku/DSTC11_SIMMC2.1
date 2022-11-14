
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import (
    PreTrainedTokenizer,
)

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer

class LineByLineDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, pred_file_path: str, target_file_path: str, block_size=1024
    ):
        assert os.path.isfile(pred_file_path)
        assert os.path.isfile(target_file_path)

        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        print("Creating features from dataset file at %s", pred_file_path)
        print("Creating features from dataset file at %s", target_file_path)

        with open(pred_file_path, encoding="utf-8") as f:
            pred_lines = [
                line.strip()
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]

        with open(target_file_path, encoding="utf-8") as f:
            target_lines = [
                line.strip()
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]

        input_ = tokenizer.batch_encode_plus(
            pred_lines, add_special_tokens=True, max_length=block_size, truncation=True
        )
        output_ = tokenizer.batch_encode_plus(
            target_lines, add_special_tokens=True, max_length=block_size, truncation=True
        )
        
        self.src = input_["input_ids"]
        self.src_mask = input_["attention_mask"]
        self.tgt = output_["input_ids"]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return torch.tensor(self.src[i], dtype=torch.long), torch.tensor(self.src_mask[i], dtype=torch.long), torch.tensor(self.tgt[i], dtype=torch.long)

class PretrainDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, pred_file_path: str, target_file_path: str, block_size: int
    ):
        assert os.path.isfile(pred_file_path)
        assert os.path.isfile(target_file_path)

        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        print("Creating features from dataset file at %s", pred_file_path)
        print("Creating features from dataset file at %s", target_file_path)
        
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        with open(pred_file_path, encoding="utf-8") as f:
            pred_lines = [
                line.strip()
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]

        with open(target_file_path, encoding="utf-8") as f:
            target_lines = [
                line.strip()
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        
        self.pred_lines = pred_lines
        self.target_lines = target_lines
        
        input_ = tokenizer.batch_encode_plus(
            pred_lines, add_special_tokens=True, max_length=block_size, truncation=True
        )
        output_ = tokenizer.batch_encode_plus(
            target_lines, add_special_tokens=True, max_length=block_size, truncation=True
        )
        
        mask_prob = 0.5 ## 0.5
        
        self.src = input_["input_ids"]
        self.src_mask = input_["attention_mask"]
        self.tgt = output_["input_ids"]
        self.tgt_mask = output_["attention_mask"]

        masked_pred_lines = [self.pos_masking(line,mask_prob) for line in tqdm(pred_lines)]
        masked_target_lines = [self.pos_masking(line,mask_prob) for line in tqdm(target_lines)]

        masked_input_ = tokenizer.batch_encode_plus(
            masked_pred_lines, add_special_tokens=True, max_length=block_size, truncation=True
        )
        masked_output_ = tokenizer.batch_encode_plus(
            masked_target_lines, add_special_tokens=True, max_length=block_size, truncation=True
        )
        
        self.masked_src = masked_input_["input_ids"]
        self.masked_src_mask = masked_input_["attention_mask"]
        self.masked_tgt = masked_output_["input_ids"]
        self.masked_tgt_mask = masked_output_["attention_mask"]
        
    def pos_masking(self, line, mask_probability): 
        nltk_tokens = nltk.word_tokenize(line)
        nltk_pos = nltk.pos_tag(nltk_tokens)
#         pos_prob = [mask_probability if nltk_pos[i][1][0]=="V" else 0 for i in range(len(nltk_tokens))]
        pos_prob = [mask_probability if nltk_pos[i][1][0]=="N" else 0 for i in range(len(nltk_tokens))]
#         pos_prob = [mask_probability if (nltk_pos[i][1][0]=="N" or nltk_pos[i][1][0]=="V") else 0 for i in range(len(nltk_tokens))]

        pos_mask = torch.bernoulli(torch.tensor(pos_prob,dtype=float)) 
        masked_indices = [nltk_tokens[i] if pos_mask[i]==0 else self.tokenizer.mask_token for i in range(len(nltk_tokens))]
        masked_line = TreebankWordDetokenizer().detokenize(masked_indices)
        return masked_line    

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        
        return (torch.tensor(self.src[i], dtype=torch.long), 
            torch.tensor(self.src_mask[i], dtype=torch.long), 
            torch.tensor(self.tgt[i], dtype=torch.long),
            torch.tensor(self.tgt_mask[i], dtype=torch.long),
            torch.tensor(self.masked_src[i], dtype=torch.long), 
            torch.tensor(self.masked_src_mask[i], dtype=torch.long), 
            torch.tensor(self.masked_tgt[i], dtype=torch.long),
            torch.tensor(self.masked_tgt_mask[i], dtype=torch.long))       

def load_and_cache_examples(args, tokenizer, evaluate=False, pretrain=False):
    pred_file_path = args.eval_data_pred_file if evaluate else args.train_data_pred_file
    target_file_path = args.eval_data_target_file if evaluate else args.train_data_target_file

    if pretrain:
        dataset = PretrainDataset(
            tokenizer, args, pred_file_path=pred_file_path, target_file_path=target_file_path, block_size=args.block_size
        )
    else:
        dataset = LineByLineDataset(
            tokenizer, args, pred_file_path=pred_file_path, target_file_path=target_file_path, block_size=args.block_size
        ) 

    # Unknown issues have been reported around not being able to handle incomplete batches (e.g. w/ older CUDA 9.2)
    # Below is a workaround in case you encounter this issue.
    # Alternatively, --nocuda could avoid this issue too.
    # Comment out the following if you do not encounuter this issue or if you are not using any GPU.
    # n = len(dataset) % args.per_gpu_train_batch_size
    # if n != 0:
    #     print("Truncating from %d examples" % len(dataset.src))
    #     dataset.src = dataset.src[:-n]
    #     dataset.src_mask = dataset.src_mask[:-n]
    #     dataset.tgt = dataset.tgt[:-n]
    #     print("Truncating to %d examples" % len(dataset.src))
    return dataset