#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

    Script for converting the main SIMMC datasets (.JSON format)
    into the line-by-line stringified format (and back).

    The reformatted data is used as input for the GPT-2 based
    DST model baseline.
"""
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


def convert_json_to_flattened(
    input_path_json,
    output_path_predict,
    output_path_target,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=True,
    input_path_special_tokens="",
    output_path_special_tokens="",
):
    """
    Input: JSON representation of the dialogs
    Output: line-by-line stringified representation of each turn
    """

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

    if output_path_special_tokens != "":
        # If a new output path for special tokens is given,
        # we track new OOVs
        oov = set()


#     scene_paths = glob("../data/simmc2_scene_jsons_dstc10_public/*_scene.json")
    scene_paths = glob("../data/public/*_scene.json")
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
    with open("../data/pred_result_path.json", mode="r") as inp:
        visual_meta = json.load(inp)
        for key in visual_meta.keys():
            new_visual_meta[key]={}
            for item in visual_meta[key]:
                new_visual_meta[key][item["index"]]=item
    visual_meta=new_visual_meta

        
    for _, dialog in enumerate(data):
        # 현재 대화에 필요한 meta data 저장 완료
        domain = dialog['domain']
        scene_ids = [scene_id for scene_id in dialog['scene_ids'].values()]        
        scene_path = [path for path in scene_paths for ids in scene_ids if ids in path]
        # visual + non visual 위치

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
            user_belief = turn[FIELDNAME_BELIEF_STATE]
            asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()

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

            # Format belief state
            if use_belief_states:
                belief_state = []
                # for bs_per_frame in user_belief:
                str_belief_state_per_frame = (
                    "{act} {SEP_1_TOKEN} [ {slot_values} ]".format(
                        act=user_belief["act"].strip(),
                        SEP_1_TOKEN = SEP_1_TOKEN,
                        slot_values=", ".join(
                            [
                                f"{k.strip()} = {str(v).strip()}"
                                for k, v in user_belief["act_attributes"][
                                    "slot_values"
                                ].items()
                            ]
                        ),
                    )
                )
                belief_state.append(str_belief_state_per_frame)

                # Track OOVs
                if output_path_special_tokens != "":
                    oov.add(user_belief["act"])
                    for slot_name in user_belief["act_attributes"]["slot_values"]:
                        oov.add(str(slot_name))
                        # slot_name, slot_value = kv[0].strip(), kv[1].strip()
                        # oov.add(slot_name)
                        # oov.add(slot_value)

                str_belief_state = " ".join(belief_state)

                # Format the main input
                predict = TEMPLATE_PREDICT.format(
                    context=context,
                    START_BELIEF_STATE=START_BELIEF_STATE,
                )
                predicts.append(predict)

                # Format the main output
                target = TEMPLATE_TARGET.format(
                    CLS_TOKEN=CLS_TOKEN,
                    belief_state=str_belief_state,
                    SEP_2_TOKEN=SEP_2_TOKEN,
                    response=asst_uttr,
                    END_TOKEN=END_TOKEN,
                )
                targets.append(target)
            else:
                # Format the main input
                predict = TEMPLATE_PREDICT_NOBELIEF.format(
                    context=context, START_OF_RESPONSE=START_OF_RESPONSE
                )
                predicts.append(predict)

                # Format the main output
                target = TEMPLATE_TARGET_NOBELIEF.format(
                    response=asst_uttr,
                    END_OF_SENTENCE=END_OF_SENTENCE,
                    START_OF_RESPONSE=START_OF_RESPONSE,
                )
                targets.append(target)

    # Create a directory if it does not exist
    directory = os.path.dirname(output_path_predict)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    directory = os.path.dirname(output_path_target)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Output into text files
    with open(output_path_predict, "w") as f_predict:
        X = "\n".join(predicts)
        f_predict.write(X)

    with open(output_path_target, "w") as f_target:
        Y = "\n".join(targets)
        f_target.write(Y)

    if output_path_special_tokens != "":
        # Create a directory if it does not exist
        directory = os.path.dirname(output_path_special_tokens)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(output_path_special_tokens, "w") as f_special_tokens:
            # Add oov's (acts and slot names, etc.) to special tokens as well
            special_tokens["additional_special_tokens"].extend(list(oov))
            json.dump(special_tokens, f_special_tokens)


def represent_visual_objects(object_ids):
    # Stringify visual objects (JSON)
    """
    target_attributes = ['pos', 'color', 'type', 'class_name', 'decor_style']

    list_str_objects = []
    for obj_name, obj in visual_objects.items():
        s = obj_name + ' :'
        for target_attribute in target_attributes:
            if target_attribute in obj:
                target_value = obj.get(target_attribute)
                if target_value == '' or target_value == []:
                    pass
                else:
                    s += f' {target_attribute} {str(target_value)}'
        list_str_objects.append(s)

    str_objects = ' '.join(list_str_objects)
    """
    str_objects = ", ".join([str(o) for o in object_ids])
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"


def parse_flattened_results_from_file(path):
    results = []
    with open(path, "r") as f_in:
        for line in f_in:
            parsed = parse_flattened_result(line)
            results.append(parsed)

    return results


def parse_flattened_result(to_parse):
    """
    Parse out the belief state from the raw text.
    Return an empty list if the belief state can't be parsed

    Input:
    - A single <str> of flattened result
      e.g. 'User: Show me something else => Belief State : DA:REQUEST ...'

    Output:
    - Parsed result in a JSON format, where the format is:
        [
            {
                'act': <str>  # e.g. 'DA:REQUEST',
                'slots': [
                    <str> slot_name,
                    <str> slot_value
                ]
            }, ...  # End of a frame
        ]  # End of a dialog
    """
    dialog_act_regex = re.compile(
        r"([\w:?.?]*)  *\[([^\]]*)\]"
    )
    slot_regex = re.compile(r"([A-Za-z0-9_.-:]*)  *= ([^,]*)")
    request_regex = re.compile(r"([A-Za-z0-9_.-:]+)")
    object_regex = re.compile(r"([A-Za-z0-9]+)")

    belief = []

    # Parse
    splits = to_parse.strip().split(SEP_2_TOKEN)
    if len(splits) == 1: # 옳바른 응답을 생성하지 못한 경우
                         # ['<pad><cls>. <end>']
        d = {
            "act": "",
            "slots": [],
            "request_slots": [0],
            "objects": [0],
        }
        belief.append(d)
        return belief        
    last_uttr = splits[1].strip().replace(END_TOKEN, "").strip()
    if len(splits) == 2:
        to_parse = splits[0].strip().replace(CLS_TOKEN, "").strip()
        splits = to_parse.replace(SEP_1_TOKEN,"").replace("  ", " ")
        for dialog_act in dialog_act_regex.finditer(splits):
            d = {
                "act": dialog_act.group(1),
                "slots": [],
                "request_slots": [0],
                "objects": [0],
            }

            for slot in slot_regex.finditer(dialog_act.group(2)):
                d["slots"].append([slot.group(1).strip(), slot.group(2).strip()])                    


            if d != {}:
                belief.append(d)

    return belief
