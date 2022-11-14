# combine results for subtask 1, 2, 3

import json

disamb_data=       # file 
coref_data=          # file 
dst_data=          # file 
output_file = "subtask123_results.json"

assert len(dst_data["dialogue_data"])==len(disamb_data["dialogue_data"])
assert len(dst_data["dialogue_data"])==len(coref_data["dialogue_data"])

for i, dialouge in enumerate(dst_data["dialogue_data"]):
    
    for j,turn in enumerate(dialogue["dialogue"]):
        
        dst_data["dialogue_data"][i]["dialogue"][j]["transcript_annotated"]["act_attributes"]["objects"]=[int(x) for x in coref_data["dialogue_data"][i]["dialogue"][j]["transcript_annotated"]["act_attributes"]["objects"]]
        dst_data["dialogue_data"][i]["dialogue"][j]["transcript_annotated"]["disambiguation_candidates"]=[int(x) for x in  disamb_data["dialogue_data"][i]["dialogue"][j]["transcript_annotated"]["disambiguation_candidates"]]
        
with open(output_file,"w") as f:
    json.dump(dst_data,f)
        

       