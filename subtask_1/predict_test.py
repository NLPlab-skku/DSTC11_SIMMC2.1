import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption
import pickle
import numpy as np
import pdb
from pathlib import Path
from torchvision import transforms
from PIL import Image
import ruamel.yaml as yaml

from tqdm import tqdm
from models.model_simmc import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

############# Hyper or Path ###############

threshold = 0.5

model_path = './output/SIMMC_base/checkpoint_best.pth'

test_file = "./data_simmc/simmc2.1_dials_dstc11_teststd_public.json"
object_file = "./data_simmc/preprocess/object_test.json"
image_folder = "./data_simmc/test_image"
result_path = "./results/simmc2.1_dials_dstc11_teststd_public.json"

config_path = "./configs/SIMMC.yaml"
save_config_path = "./pred/SIMMC"
device = torch.device('cuda')

############################################

config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

Path(save_config_path).mkdir(parents=True, exist_ok=True)
Path("./results").mkdir(parents=True, exist_ok=True)

yaml.dump(config, open(os.path.join(save_config_path, 'config.yaml'), 'w'))

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

test_transform = transforms.Compose([
    transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
    ])   

image_root = config['image_root']

n_turns = 2

# device = torch.device('cuda')

MENTIONED_OBJECT = "<MO>"
NOT_MENTIONED_OBJECT = "<NMO>"
USER_TURN = "<UT>"
SYSTEM_TURN = "<ST>"

with open(object_file) as f:
    obejct_data = json.load(f)

with open(test_file) as f:
    test_data = json.load(f)

special_tokens_dict= {"additional_special_tokens": ["<MO>", "<NMO>", "<UT>", "<ST>"]}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_special_tokens(special_tokens_dict)

# model, tokenizer load
model = ALBEF(config=config, text_encoder='bert-base-uncased', tokenizer=tokenizer)
model.text_encoder.resize_token_embeddings(len(tokenizer))
if config['distill']:
    model.text_encoder_m.resize_token_embeddings(len(tokenizer))
    
####### 체크 포인트 설정
 
checkpoint = torch.load(model_path, map_location='cpu') 
state_dict = checkpoint['model']

# reshape positional embedding to accomodate for image resolution change
pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)         
state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
msg = model.load_state_dict(state_dict, strict=False)
print('load model from %s'%checkpoint)
print(msg)


model = model.to(device)   
model.eval()

for test_data_i, object_info in tqdm(zip(test_data['dialogue_data'], obejct_data)): # 발화
    data_i = test_data_i["dialogue"]
    n = len(data_i)
    scenes = list(test_data_i["scene_ids"].values())
    point = list(test_data_i["scene_ids"].keys())[-1]    
    
    for idx in range(n): # Turn
        
        dial_list = []
        
        for i in range(min(n_turns - 1, idx)):
            dial_list.append(USER_TURN + " " + data_i[idx + 1 - n_turns - i]['transcript'])
            dial_list.append(SYSTEM_TURN + " " + data_i[idx + 1 - n_turns - i]['system_transcript'])
        
        dial_list.append(USER_TURN + " " + data_i[idx]['transcript'])
        
        dial_i = " ".join(dial_list)
        # label_i = data_i['labels'][idx]
        mentioned_list = object_info['mentioned_object'][idx]

        pred_object_list = []
        
        for object_id in object_info['meta'].keys():
            object_meta = object_info['meta'][object_id]
                
            if int(object_id) in mentioned_list:
                is_mentioned = MENTIONED_OBJECT
            else:
                is_mentioned = NOT_MENTIONED_OBJECT
                
            meta_keys = sorted(list(object_meta.keys()))

            if "availableSizes" in meta_keys:
                meta_keys.remove("availableSizes")

            # Remove Visual Meta data
            if "color" in meta_keys:
                meta_keys.remove("color")
            if "pattern" in meta_keys:
                meta_keys.remove("pattern")
            if "type" in meta_keys:
                meta_keys.remove("type")
            if "sleeveLength" in meta_keys:
                meta_keys.remove("sleeveLength")
            if "assetType" in meta_keys:
                meta_keys.remove("assetType")

            meta_list = [str(meta_key) + " is " + str(object_meta[meta_key]) + "." for meta_key in meta_keys]
            meta_str = " ".join(meta_list)
            
            # dial + meta + label + object_id + empty_label
            # 이전 
            if int(point)==0 or int(idx)<int(point):
                if os.path.exists(f"{image_folder}/{scenes[0]}_{object_id}.jpeg"):
                    model_input = [dial_i, meta_str, object_id, is_mentioned, f"{image_folder}/{scenes[0]}_{object_id}.jpeg"]
                else:
                    model_input = [dial_i, meta_str, object_id, is_mentioned, f"{image_folder}/fallback.jpeg"]
            else:
                if os.path.exists(f"{image_folder}/{scenes[1]}_{object_id}.jpeg"):
                    model_input = [dial_i, meta_str, object_id, is_mentioned, f"{image_folder}/{scenes[1]}_{object_id}.jpeg"]
                else:
                    model_input = [dial_i, meta_str, object_id, is_mentioned, f"{image_folder}/fallback.jpeg"]

            image_path = model_input[-1]

            model_input[0] += (' ' + model_input[1])
            
            image = Image.open(image_path).convert('RGB')   
            image = test_transform(image).unsqueeze(0) # add batch dim

            sentence = pre_caption(model_input[0], 200) # max words = 200

            prev_mention_token = model_input[-2]

            text = [" ".join([sentence,prev_mention_token])]
            images= image.to(device, non_blocking=True)

            text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device) 

            prediction = model(images, text_inputs, targets=None, train=False)[0]
            prob_i = torch.nn.functional.softmax(prediction)[1].item()

            pred = prob_i > threshold

            if pred:
                pred_object_list.append(int(object_id))
            
            #_, pred_class = prediction.max(1)

        print(pred_object_list)
        data_i[idx]['transcript_annotated'] = {
                        "disambiguation_candidates": pred_object_list
                    }

            # pred_class가 어떤 형태인지 보고 pred_object_list 생성
            # data.append(data_str)


with open(result_path, "w") as f:
    json.dump(test_data, f, indent=4)
