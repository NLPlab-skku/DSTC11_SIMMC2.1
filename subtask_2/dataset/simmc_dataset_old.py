import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption
import pickle
import numpy as np

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
        with (open("data/clip_resnext.pickle", "rb")) as openfile:
            objects=pickle.load(openfile)

        for k in objects.keys():
            for kk in objects[k].keys():
                for kkk in objects[k][kk].keys():
                    if kkk != "prefab_path":
#                         print(objects[k][kk][kkk])
                        objects[k][kk][kkk]=objects[k][kk][kkk].detach().cpu().numpy()
        print("no error in dataset building")

#         self.clip= [o["clip"].tolist() for o in objects]
#         self.rcnn= [o["rcnn"].tolist() for o in objects]
#         self.resnext= [o["rexnext"].tolist() for o in objects]
            
        self.objects = objects
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index].split("\t")
        
        image_path = ann[-1].strip()
        # image_path example: m_cloth_store_1416238_woman_9_2_35.jpeg
        scene , index = parse_path(image_path)
        object_embeds = self.objects[scene][index]
        
#         clip = [1] * 512
#         rcnn = [1] * 2048
#         resnext = [1] * 2048

        clip = object_embeds["clip"]
        rcnn = object_embeds["rcnn"]
        resnext = object_embeds["resnext"]
        prefabpath = object_embeds["prefab_path"]
        
        image = Image.open("data_simmc/"+image_path).convert('RGB')   
        image = self.transform(image)          

        #meta_data = ann[1]
        sentence.append(meta_data)
        sentence = pre_caption(ann[0], self.max_words)
        label = ann[2]
        prev_mention_token = ann[-2]
        
        
        return image, sentence, self.labels[label], prev_mention_token, clip, rcnn, resnext # prev mention token
    