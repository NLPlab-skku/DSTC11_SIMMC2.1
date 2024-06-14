# _DSTC11 Track1 SIMMC 2.1_
**This work is a result of the cooperation of SungKyunKwan-University NLP Lab and LG Electronics**\
**Lab Home: https://nlplab-skku.github.io/** 

## Overview
- We used different models for different subtasks
As we trained each subtask in different environments, they each have their own data folder.
**We got Third Place on sub task 2 (Coreferencing), sub task 3 (Dialogue State Tracking) and sub task 4 (Response Generation)**

### Subtask1 Disambiguation
We have one submissions for subtask1 which train a vision language transformer model using ALBEF as backbone.


### Subtask2 Coreferencing
Similar to subtask1, we train a vision language transformer model using ALBEF as backbone.


### Subtask3 Dialogue State Tracking
We used additional pretraining on the BART model with a tailored masked language modeling task to adapt to the dialogue domain, later finetuned on the Dialogue State Tracking task.


### Subtask4 Response Generation
Similar to subtask 3, We used additional pretraining on the BART model with a tailored masked language modeling task to adapt to the dialogue domain, later finetuned on the Response Generation.


---
## Enviroment
We use different environments for different subtasks.
Check the requirements.txt for each subtask.


---
## Model Parameters
Model Parameters needed for each subtask is provided by a google drive link
Check the README.md for the model of the subtask


## Image Preprocessing
We preprocessed the object images using a ResNext model.
The details are in the image_model folder.



---
## Final result (we submitted)
The final prediction files that we submitted are located in the path below each branch.
**These are the final prediction results intended to be compared to those of other models**
Each of the submitted file can be found below

### sub task 1 (json)
/subtask_1/results/dstc11-simmc-teststd-pred-subtask-123.json

### sub task 2 (json)
/subtask_2/results/dstc11-simmc-teststd-pred-subtask-123.json

### sub task 3 (json)
/subtask_3_4/bart_dst_response/results/dstc11-simmc-teststd-pred-subtask-123.json

### sub task 4 
/subtask_3_4/bart_dst_response/results/dstc11-simmc-teststd-pred-subtask-4.json

### Devtest Results(updated! we had a problem with evaluation)
![image](https://user-images.githubusercontent.com/38829593/198830052-c2d6d546-c3c8-4fbb-80ad-f054839e3741.png)




