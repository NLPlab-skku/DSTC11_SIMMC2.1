import os
import json
import yaml
import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm
import traceback

def build_image_data(args):
    # Path    
    if args.data == "fashion":
        meta_path = "./data/fashion_prefab_metadata_all.json"
        image_path = "./data/fashion/" + args.image_path + "/train"
        
        if os.path.isdir("./data/fashion/" + args.image_path):
            pass
        else:
            os.mkdir("./data/fashion/" + args.image_path)
            
        if os.path.isdir(image_path):
                pass
        else:
            os.mkdir(image_path)
        
    else:
        meta_path = "./data/furniture_prefab_metadata_all.json"
        image_path = "./data/furniture/" + args.image_path + "/train"

        if os.path.isdir("./data/furniture/" + args.image_path):
            pass
        else:
            os.mkdir("./data/furniture/" + args.image_path)
            
        if os.path.isdir(image_path):
                pass
        else:
            os.mkdir(image_path)

    # Load metadata
    with open(meta_path) as f:
        data = json.load(f)

    # Load Filtering_Config
    with open(args.image_filtering) as f:
        filtering_config = yaml.load(f)

    # Make directory
    for data_i in data.keys():
        class_name = data_i.replace("/", "_")
        if os.path.isdir("{0}/{1}".format(image_path, class_name)):
            pass
        else:
            os.mkdir("{0}/{1}".format(image_path, class_name))
            
    ## Load scene, bbox data
    scene_folder = "./data/public"
    scenes = sorted(glob(str(scene_folder + "/*_scene.json")))

    ## image file list
    image_path1 = "./data/simmc2_scene_images_dstc10_public_part2"
    image_path2 = "./data/simmc2_scene_images_dstc10_public_part1"
    image_files1 = [os.path.join(image_path1, f) for f in os.listdir(image_path1) if os.path.isfile(os.path.join(image_path1, f))]
    image_files2 = [os.path.join(image_path2, f) for f in os.listdir(image_path2) if os.path.isfile(os.path.join(image_path2, f))]
    image_path = image_files1 + image_files2
    image_path_dict = {file_i.split("/")[3]: file_i for file_i in image_path}
            
    n_data = 0

    for scene in tqdm(scenes):
        with open(scene) as f:
            scene_data = json.load(f)
        scene_objects = scene_data["scenes"][0]["objects"] #object lists

        image_file = image_path_dict[(scene.replace("m_", "")).split("/")[3].replace("_scene.json", ".png")]
        image_i = Image.open(image_file)

        for object_i in scene_objects:
            prefab_path = object_i["prefab_path"]
            bbox = object_i["bbox"]

            if prefab_path in data.keys():
                try:
                    if filtering(bbox, filtering_config):
                        ## Valid bbox
                        croppedImage = image_i.crop((bbox[0], bbox[1], bbox[0] + bbox[3], bbox[1] + bbox[2]))

                        class_name = prefab_path.replace("/", "_")
                        croppedImage.save("{0}/{1}/{2}.png".format(image_path, class_name, str(n_data)))
                        n_data += 1
                            
                except Exception as e:
                    print("Crop Error : ", scene, prefab_path, bbox)
                    print(traceback.format_exc())

def filtering(bbox, filtering_config):
    result = True
    min_pixel = filtering_config.Train.minPixel
    
    if min_pixel > 0:
        if bbox[2] <= min_pixel and bbox[3] <= min_pixel:
            result = False
        
    return result
         
def main():
    parser = argparse.ArgumentParser()

    # Transform Method
    parser.add_argument("--data", type=str, default="fashion", help="Data Type")
    parser.add_argument("--image_path", type=str, help="Image Folder Path")
    parser.add_argument("--image_preprocessing", type=str, default=None, help="Image Preprocessing Method")
    parser.add_argument("--image_filtering", type=str, default=None, help="Image Filtering Method")

    args = parser.parse_args()
    
    build_image_data(args)

if __name__ == "__main__":
    main()