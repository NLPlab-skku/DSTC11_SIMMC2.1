#!/usr/bin/env python3
"""
Inference for subtask 4, dialogue generation
"""

import torch
from model.bart_mlm import BartForMaskedLM
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
)

import argparse
import json
import re
import os
from glob import glob
# DSTC style dataset fieldnames
FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_BELIEF_STATE = "=> Belief State :"
# START_OF_RESPONSE = "<SOR>"
# END_OF_BELIEF = "<EOB>"
# END_OF_SENTENCE = "<EOS>"

CLS_TOKEN = '<cls>'
SEP_1_TOKEN = '<sep1>'
SEP_2_TOKEN = '<sep2>'
END_TOKEN = '<end>'
TEMPLATE_PREDICT = "{context} {START_BELIEF_STATE} "
TEMPLATE_TARGET = (
    "{CLS_TOKEN} {belief_state} {SEP_2_TOKEN} {response} {END_TOKEN}"
)

# No belief state predictions and target.
TEMPLATE_PREDICT_NOBELIEF = "{context} {START_OF_RESPONSE} "
TEMPLATE_TARGET_NOBELIEF = "{START_OF_RESPONSE} {response} {END_OF_SENTENCE}"

def object_id_to_meta(objects, meta_data):
    total_obj_data = list()
    each_obj_data = list()
    for obj in objects:
        # 1
        each_obj_data.append(str(obj))

        meta = list()
        for key, value in meta_data[obj].items():
#             meta.append("{}:{}".format(key,value))
            meta.append("{}".format(value))

        # color:red, pattern:stripped, type:skirt
        meta = (", ").join(meta)
        # color: red, pattern: stripped, type: skirt
        meta = meta.replace(":", ": ")

        each_obj_data.extend(['{', meta, '}'])
        total_obj_data.append((" ").join(each_obj_data))
        each_obj_data.clear()
    total_obj_data = (", ").join(total_obj_data)
    return total_obj_data


def inference_from_json(
    args,
    input_path_json,
    output_path_json,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=True,
    input_path_special_tokens="",
):
    """
    Input: JSON representation of the dialogs
    Output: Inferenced output for subtask 4
    """
    
    args.device = torch.device("cuda")
    
    # prepare data
    with open(input_path_json, "r") as f_in:
        data = json.load(f_in)["dialogue_data"]

    predicts = []
    targets = []
    if input_path_special_tokens != "":
        with open(input_path_special_tokens, "r") as f_in:
            special_tokens = json.load(f_in)
    else:
        special_tokens = {"eos_token": END_TOKEN}
        additional_special_tokens = []
        if use_multimodal_contexts:
            additional_special_tokens.extend(
                [START_OF_MULTIMODAL_CONTEXTS, END_OF_MULTIMODAL_CONTEXTS]
            )
        additional_special_tokens.extend([CLS_TOKEN, SEP_1_TOKEN, SEP_2_TOKEN, END_TOKEN, '<unk>', '<pad>', '<cst>'])
        special_tokens["additional_special_tokens"] = additional_special_tokens
    
    # set scene path
    scene_paths = glob("../data/simmc2_scene_jsons_dstc10_teststd/*_scene.json")
#     scene_paths = glob("../data/public/*_scene.json")
    
    # prepare meta data
    new_meta_fashion = {}
    new_meta_furniture={}
    new_visual_meta={}
    with open("../data/fashion_prefab_metadata_all.json", mode="r") as inp:
        meta_fashion = json.load(inp)
        for key in meta_fashion.keys():
            new_key = key.replace("/","_")
            new_meta_fashion[new_key]=meta_fashion[key]
    with open("../data/furniture_prefab_metadata_all.json", mode="r") as inp:
        meta_furniture = json.load(inp)
        for key in meta_furniture.keys():
            new_key = key.replace("/","_")
            new_meta_furniture[new_key]=meta_furniture[key]
#     with open("../data/pred_result_path.json", mode="r") as inp:
    with open("../data/pred_test_result.json", mode="r") as inp:
        visual_meta = json.load(inp)
        for key in visual_meta.keys():
            new_key=key.split("/")[-1].rstrip("_scene.json")
            new_visual_meta[new_key]={}
            for item in visual_meta[key]:
                new_visual_meta[new_key][item["index"]]=item
    visual_meta=new_visual_meta

    # PREPARE MODEL
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)    
    model = BartForMaskedLM.from_pretrained(args.model_name_or_path)
    model.to(torch.device("cuda")) 
    
    
    # Save dictionary
    result_list=[]
    
    p_index = 0
    
    # build input for each instance
    for _, dialog in enumerate(data):
        # 현재 대화에 필요한 meta data 저장 완료
        domain = dialog['domain']
        scene_ids = [scene_id for scene_id in dialog['scene_ids'].values()]        
        scene_path = [path for path in scene_paths for ids in scene_ids if ids in path]
        
        # build return_dict
        result_dict = {"dialog_id": dialog["dialogue_idx"]}
        result_dict["predictions"] = []

        
        # prepare metadata for scene
        meta_data = dict()
        for s_path in scene_path:
            s_name = s_path.split("/")[-1].rstrip("_scene.json")
            with open(s_path, mode="r") as inp:
                data = json.load(inp)['scenes'][0]['objects']
                for obj in data:
                    obj_name = obj["prefab_path"]
                    obj_idx = obj["index"]

                    meta_data[obj_idx] = dict()
                    if domain == 'fashion': 
                        meta_data[obj_idx]['customerReview'] = meta_fashion[obj_name]['customerReview']
                        meta_data[obj_idx]['brand'] = meta_fashion[obj_name]['brand']
                        meta_data[obj_idx]['size'] =meta_fashion[obj_name]['size']
                        meta_data[obj_idx]['price'] = meta_fashion[obj_name]['price']
                        if "pred_path" in visual_meta[s_name][obj_idx]:
                            meta_data[obj_idx]['type'] = new_meta_fashion[visual_meta[s_name][obj_idx]["pred_path"]]["type"]   # Fashion
                            meta_data[obj_idx]['pattern'] = new_meta_fashion[visual_meta[s_name][obj_idx]["pred_path"]]["pattern"]                
                            meta_data[obj_idx]['color'] = new_meta_fashion[visual_meta[s_name][obj_idx]["pred_path"]]["color"] 

                    else:                                         
                        meta_data[obj_idx]['customerRating'] = meta_furniture[obj_name]['customerRating']                        
                        meta_data[obj_idx]['brand'] = meta_furniture[obj_name]['brand']                        
                        meta_data[obj_idx]['price'] = meta_furniture[obj_name]['price']
                        if "pred_path" in visual_meta[s_name][obj_idx]:
                            meta_data[obj_idx]['type'] = new_meta_furniture[visual_meta[s_name][obj_idx]["pred_path"]]["type"]    # Furniture
                            meta_data[obj_idx]['materials'] =  new_meta_furniture[visual_meta[s_name][obj_idx]["pred_path"]]["materials"]    
                            meta_data[obj_idx]['color'] =  new_meta_furniture[visual_meta[s_name][obj_idx]["pred_path"]]["color"]    

        prev_asst_uttr = None
        prev_turn = None
        lst_context = []
        prev_objects_memory = list()

        for turn in dialog[FIELDNAME_DIALOG]:
            user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()


            # Format main input context
            context = ""
            if prev_asst_uttr:
                context += f"System : {prev_asst_uttr} "
                if use_multimodal_contexts:
                    # Add multimodal contexts
                    visual_objects = prev_turn[FIELDNAME_SYSTEM_STATE][
                        "act_attributes"
                    ]["objects"]
                    if len(visual_objects) > 0:
                        prev_objects_memory = visual_objects.copy()
                    elif len(visual_objects) == 0:
                        visual_objects = prev_objects_memory.copy()

                    visual_objects = [object_id_to_meta(visual_objects, meta_data)]                                        
                    context += represent_visual_objects(visual_objects) + " "
            context += f"User : {user_uttr}"
            if FIELDNAME_ASST_UTTR in turn:
                asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()
                prev_asst_uttr = asst_uttr
            prev_turn = turn

            # Add multimodal contexts -- user shouldn't have access to ground-truth
            """
            if use_multimodal_contexts:
                visual_objects = turn[FIELDNAME_BELIEF_STATE]['act_attributes']['objects']
                context += ' ' + represent_visual_objects(visual_objects)
            """

            # Concat with previous contexts
            lst_context.append(context)
            context = " ".join(lst_context[-len_context:])

            # Format the main input
            predict = TEMPLATE_PREDICT.format(
                context=context,
                START_BELIEF_STATE=START_BELIEF_STATE,
            )
            
            # predict를 모델에 넣을 것.
            encoded_prompt = tokenizer(
                [predict], add_special_tokens=True, max_length=1024,truncation=True)            
            src = torch.tensor(encoded_prompt.input_ids).to(args.device)
            src_mask = torch.tensor(encoded_prompt.attention_mask).to(args.device)
            output_sequences = model.generate(
                src,
                max_length=100 + len(src),
                decoder_start_token_id=tokenizer.pad_token_id,
                attention_mask=src_mask,
                early_stopping=True)
            
            p_index+=1  
            generated_sequences = []            
            for generated_sequence_idx, generated_sequence in enumerate(
                output_sequences
            ):
                print(
                    "=== GENERATED SEQUENCE {sequence_idx}, {promt_idx}/{n_prompts} ===".format(
                        sequence_idx=generated_sequence_idx,
                        promt_idx=p_index,
                        n_prompts=8609,
                    )
                )
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(
                    generated_sequence, clean_up_tokenization_spaces=True
                )
                text = text[: text.find("<end>")] # stop token
                text =text.replace("<pad>", "").replace("<s>", "")
                
                # subtask4 preprocessing for only response
                
                split_line = text.split(SEP_2_TOKEN)
                if len(split_line) == 1:       # split_line 길이가 1이라는 건, EOB 토큰이 없다는 거고 생성 실패했다는 소리
                    print("split_line : {}".format(split_line))            
                    res_text = split_line[0].strip("\n").replace("<cls>","").replace("<pad>","").replace("<end>","").lstrip().rstrip()
                else:
                    prompt = split_line[0].strip("\n").strip(" ")
                    res_text = split_line[1].strip("\n").replace("<cls>","").replace("<pad>","").replace("<end>","").lstrip().rstrip()
                
                print("2 TEXT : {}".format(res_text))
                generated_sequences.append(res_text)
        
            # build turn dict
            turn_dict={"turn_id":turn["turn_idx"]}

            # response만 뽑히도록 수정해야함.
            turn_dict["response"] = generated_sequences[0]

            result_dict["predictions"].append(turn_dict)
            
        result_list.append(result_dict)
         
#         import pdb
#         pdb.set_trace()
            
            
            # json 파일 열어서 거기에 추가해줄것.
                
    
    
    # Create a directory if it does not exist
    directory = os.path.dirname(output_path_json)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Output into text files
    with open(output_path_json, "w") as f_output:
        json.dump(result_list,f_output)


def represent_visual_objects(object_ids):
    # Stringify visual objects (JSON)
    str_objects = ", ".join([str(o) for o in object_ids])
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"


if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path_json", help="input path to the original dialog data"
    )
    parser.add_argument("--output_path_json", help="output path to the inferenced dialog data")
    parser.add_argument(
        "--input_path_special_tokens",
        help="input path for special tokens. blank if not provided",
        default="",
    )
    parser.add_argument(
        "--len_context",
        help="# of turns to include as dialog context",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--use_multimodal_contexts",
        help="determine whether to use the multimodal contexts each turn",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="only bart",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="from save file or from huggingface",
    )
    
    args = parser.parse_args()
    input_path_json = args.input_path_json
    output_path_json = args.output_path_json
    input_path_special_tokens = args.input_path_special_tokens
    len_context = args.len_context
    use_multimodal_contexts = bool(args.use_multimodal_contexts)

    inference_from_json(
        args,
        input_path_json,
        output_path_json,
        input_path_special_tokens=input_path_special_tokens,
        len_context=len_context,
        use_multimodal_contexts=use_multimodal_contexts,
    )




