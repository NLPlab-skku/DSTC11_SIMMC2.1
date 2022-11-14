Training cls model 
1. run base_fasion_object_classifier.py -> ./parameters/object_model_state_fastion.bin
2. run base_furniture_object_classifier.py -> ./parameters/object_model_state_furniture.bin

Predict path for the test std data : predict metadata label for each scene, each object and integrate to one file
1. locate data test std file in parent folder
2. run pred_path_from_data.py -> ./result/pred_result_path.json