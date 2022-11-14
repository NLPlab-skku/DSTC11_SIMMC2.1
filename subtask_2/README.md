## Subtask2

We use ALBEF as the backbone for subtask2

For the sake of compatability we left most original files untouched
The simmc dataset inside the the "data_simmc" directory, and is preprocessed within it.

To run the model, run the code below

'''sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env SIMMC.py \
--config ./configs/SIMMC.yaml \
--output_dir output/SIMMC_base \
--checkpoint ./ALBEF.pth
'''


Our trained model parameter can be found in the link in the outputs directory.

### Requirements:
* pytorch 1.8.0
* transformers 4.8.1
* timm 0.4.9

### Download:

* Pre-trained checkpoint [[14M](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth)] / [[4M](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth)]


