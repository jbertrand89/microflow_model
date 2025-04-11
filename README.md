# microflow

### create configs
python save_train_config.py --config_filename ../../data/configs/train_fault_deform/iterative_geoflownet_separated_weights_intermediatel1_noreg.yaml --amp --dataset_dir <your_directory_containing_fault_deform> --split_dir <your_directory_containing_the_split_files_for_fault_deform> 


### Train the model on FaultDeform
python -u train_fault_deform.py --train_config_name iterative_geoflownet_separated_weights_intermediatel1_noreg --checkpoints_dir <your_checkpoint_dir> --offline_dir <your_offline_dir> --save_offline --seed 1


### Inference of FaultDeform
python inference_fault_deform.py --train_config_name iterative_geoflownet_separated_weights_intermediatel1_noreg --pretrained_model_filename <your_pretrained_model> --dataset_name faultdeform --metric_filename <your_metric_filename> --split_scaling_factors $v --save_metrics --dataset_dir <your_directory_containing_faultdeform> --split_dir <your_directory_containing_the_split_files> 


### Inference for real examples
python inference_real_examples.py --config_filename <your_filename> --pretrained_model_filename <your_pretrained_model> -b 1 --dataset_dir <your_directory_with_pre_post_files> --save_dir <your_saving_directory> --window_size 256 --padding 64
