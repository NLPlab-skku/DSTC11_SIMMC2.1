## Overview
- This model is for DSTC11 Track1, subtask3, subtask4
---
## Environment

- CUDA 11.1
- Python 3.7+

Packages:
- torch==1.8.0
- transformers==4.8.0
- nltk==3.6.2
- scikit-learn==0.24.2
- tensorboard==2.6.0
- tensorboardX==2.4
- wordninja==2.0.0
---

## Run code
Go to the bart_dst_response directory
'''sh
cd bart_dst_response
'''

### Run Train
'''sh
./run_preprocess_BART.sh
./run_preprocess_pretrain_BART.sh
'''

### Run Pretrain
'''sh
./run_pretrain_BART.sh
'''


### Run Train
To train from pretrained model, change the model_path to the pretrained model
'''sh
./run_train_BART.sh
'''


### Run Evaluate
'''sh
./run_generate_BART.sh
./run_evaluate_BART.sh
./run_evaluate_response_BART.sh
'''

