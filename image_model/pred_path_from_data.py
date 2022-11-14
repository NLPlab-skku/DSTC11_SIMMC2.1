import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
from torch import nn, optim
from PIL import Image
import timm
from glob import glob
from torchvision import models
from torch.optim import lr_scheduler
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
## Load meta data

def main():
    ## Load prefab data
    with open("../data/fashion_prefab_metadata_all.json") as f:
        data_fashion = json.load(f)


    ## Load scene, bbox data
    scene_folder = "../data/simmc2_scene_jsons_dstc10_teststd"
    scenes = sorted(glob(str(scene_folder + "/*_scene.json")))
    bboxes = sorted(glob(str(scene_folder + "/*_bbox.json")))

    ## image file list
    image_path1 = "../data/simmc2_scene_images_dstc10_teststd"
    image_files1 = [os.path.join(image_path1, f) for f in os.listdir(image_path1) if os.path.isfile(os.path.join(image_path1, f))]
    image_path = image_files1
    image_path_dict = {file_i.split("/")[3]: file_i for file_i in image_path}

    n_fashion = 0
    n_furniture = 0

    output_json = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_nums = [0.274, 0.258, 0.259] # [0.485, 0.456, 0.406] -> [0.274, 0.258, 0.259]
    std_nums = [0.207, 0.200, 0.204] # [0.229, 0.224, 0.225] -> [0.207, 0.200, 0.204]

    # transform resize img
    transform = T.Compose([
    T.ToTensor(),
    T.Resize(size=[256,256]),
    T.Normalize(mean_nums, std_nums)
    ])

    # load class name from train img data just for cls name
    dataset = ImageFolder(root='./data/furniture/251_class_train')
    funiture_class_name = dataset.classes
    dataset = ImageFolder(root='./data/fashion/251_class_train')
    fashion_class_name = dataset.classes

    # initialize model for fashion
    model = models.resnext101_32x8d()
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 251)
    model.to(device)
    model.load_state_dict(torch.load(f'./parameters/object_model_state_fashion.bin'))

    #initialize model for furniture
    model_f = models.resnext101_32x8d()
    n_features = model_f.fc.in_features
    model_f.fc = nn.Linear(n_features, 30)
    model_f.to(device)
    model_f.load_state_dict(torch.load(f'./parameters/object_model_state_furniture_no.bin'))

    output_json = {}
    count = 0
    correct = 0
    wrong = 0

    for scene in tqdm(scenes):
        with open(scene) as f:
            scene_data = json.load(f)
        scene_objects = scene_data["scenes"][0]["objects"]
        image_file = image_path_dict[(scene.replace("m_", "")).split("/")[3].replace("_scene.json", ".png")]
        image_i = Image.open(image_file)

        for object_i in scene_objects:
            count+=1
            prefab_path = object_i["prefab_path"]
            index = object_i["index"]
            bbox = object_i["bbox"]
            try:
                if bbox[2] >= 16 or bbox[3] >= 16:
                    ## Valid bbox
                    if prefab_path in data_fashion.keys():
                        ## fashion data                
                        croppedImage = image_i.crop((bbox[0], bbox[1], bbox[0] + bbox[3], bbox[1] + bbox[2]))
                        object_i["pred_path"] = fashion_class_name[cls_model(transform(croppedImage.convert('RGB')),model,device)]
                    else:
                        ## furniture data
                        croppedImage = image_i.crop((bbox[0], bbox[1], bbox[0] + bbox[3], bbox[1] + bbox[2]))
                        object_i["pred_path"] = funiture_class_name[cls_model(transform(croppedImage.convert('RGB')),model_f,device)]
                    if object_i["pred_path"]==object_i["prefab_path"].replace('/','_'):
                        correct+=1
                    else:
                        wrong+=1
            except Exception as e:
                print("Crop Error ")
        output_json[scene]=scene_objects
    print(f'score : {correct/count}, correct = {correct}, wrong ={wrong}, error = {count-correct-wrong} ')
    with open("./result/pred_test_result.json", "w") as json_file:
        json.dump(output_json, json_file)


        
def cls_model(image,model,device):
    model.eval()
    with torch.no_grad():
        inputs = image.to(device).unsqueeze(dim=0)
        outputs = model(inputs)
        _, o_preds = torch.max(outputs, dim=1)
        pred_list = o_preds
    return o_preds[0].item()


if __name__=="__main__":
    main()