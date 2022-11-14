import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption
import pickle
import numpy as np
import pdb

def parse_path(string):
    string = string.replace(".jpeg","")
    string = string.split("_")
    index = int(string[-1])
    string = "_".join(string[:-1])
    string = string.split("/")[1]
    return string, index

class simmc_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=200):        
        self.ann = open(ann_file,'r').readlines()#[:1000] #1000개 테스트
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.labels = {'1':1,'0':0}
        self.samples_per_cls = [0, 0]

        for idx in range(len(self.ann)):
            ann_i = self.ann[idx].split("\t")
            label = ann_i[2]

            if label == "1":
                self.samples_per_cls[1] += 1
            else:
                self.samples_per_cls[0] += 1


    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index].split("\t")

        image_path = ann[-1].strip()

        ann[0] += (' ' + ann[1])
        
        image = Image.open("data_simmc/"+image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(ann[0], self.max_words)
        label = ann[2]
        prev_mention_token = ann[-2]

        
        return image, sentence, self.labels[label], prev_mention_token #, clip, rcnn, resnext # prev mention token
