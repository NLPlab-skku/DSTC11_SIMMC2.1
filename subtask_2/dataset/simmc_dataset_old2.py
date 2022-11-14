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
        with (open("data/pred_result_path.json", "rb")) as openfile:
            objects=json.load(openfile)

        with open("./data_simmc/fashion_prefab_metadata_all.json") as f:
            meta_data = json.load(f)

        with open("./data_simmc/furniture_prefab_metadata_all.json") as f:
            meta_data_fur = json.load(f)
            
        meta_data.update(meta_data_fur)
        # 모두 _ 로 바꿔줌
        new_meta = {}   
        for k in meta_data.keys():
            new_k = k.replace('/','_') 
            new_meta[new_k] = meta_data[k]
            
        self.meta_data = new_meta

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

        meta_data = self.meta_data
        
        image_path = ann[-1].strip()
        # image_path example: m_cloth_store_1416238_woman_9_2_35.jpeg
        scene , index = parse_path(image_path)
        object_embeds = self.objects[scene][index]

        try:
            object_meta = meta_data[object_embeds['prefab_path']]
            
            descript = '<sep> '
            object_list = []
            for k in object_meta.keys():
                if k != 'assetType':
                    tmp_str = ""
                    tmp_str += (k + ' : ')
                    tmp_str += str(object_meta[k]) 
                    object_list.append(tmp_str)
            descript += ", ".join(object_list)

                    
        except:
            print(scene)
            print(index)
            print(object_embeds['prefab_path'])
            

        ann[0] += (' ' + descript)

#         print(ann[0])

        clip = object_embeds["clip"]
        rcnn = object_embeds["rcnn"]
        resnext = object_embeds["resnext"]
        
        image = Image.open("data_simmc/"+image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(ann[0], self.max_words)
        label = ann[2]
        prev_mention_token = ann[-2]

        
        return image, sentence, self.labels[label], prev_mention_token, clip, rcnn, resnext # prev mention token