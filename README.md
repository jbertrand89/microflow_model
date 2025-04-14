# MicroFlow 

[[arXiv](https://jbertrand89.github.io/microflow/) ] [[project page](https://jbertrand89.github.io/microflow/)]

This repository contains official code for MicroFlow: Domain-Specific Optical Flow for Ground Deformation Estimation in Seismic Events.

## Installation
You can setup the environment either by loading the `requirements.txt` file or by following:

```
python -m venv ENV
source ENV/bin/activate
pip install torch, rasterio, scikit-image, matplotlib, tqdm, opencv-python, pyyaml
pip install prox-tv
```

## Training on the semi-synthetic dataset FaultDeform

### Downloading the FaultDeform dataset
The FaultDeform dataset is a synthetic dataset that can be downloaded [here](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/G02ZXZ&version=1.0).
You can download it, it will follow the following file structure:

```
FaultDeform_root_dir
├── train
│   │── 000000_0_sample.npy.npz
│   │── 000000_1_sample.npy.npz
│   │── 000000_2_sample.npy.npz
│   │── 000000_3_sample.npy.npz
│   └── ...
├── val
│   │── 000000_0_sample.npy.npz
│   │── 000000_1_sample.npy.npz
│   │── 000000_2_sample.npy.npz
│   │── 000000_3_sample.npy.npz
│   └── ...
└── test
    │── 000000_0_sample.npy.npz
    │── 000000_1_sample.npy.npz
    │── 000000_2_sample.npy.npz
    │── 000000_3_sample.npy.npz
    └── ...
```

### Training Microflow

To train MicroFlow, run
```
python -u train_fault_deform.py \
--train_config_name irseparated_geoflownet_intermediatel1_noreg \
--checkpoints_dir <your_checkpoint_dir> \
--offline_dir <your_offline_dir> \
--save_offline \
--dataset_dir <FaultDeform_root_dir> \
--split_dir <your_directory_containing_the_split_files_for_fault_deform> 
--seed 1
```
using the pre-saved config located in `data/configs/train_fault/deform`
and setting 
- `checkpoint_dir`: the directory for saving the checkpoints
- `offline_dir`: the directory for saving the wandb offline logs if you setup `--save_offline`
- `dataset_dir`: the root directory of Fault Deform
- `split_dir`: the directory containing the split files

### Pre-saved configs
Pre-saved configs for the pretrained models can be found in `data/configs/train_fault/deform`.

To create a new config, use the code in `src/configs/save_train_config.py`.
Fow example, to create the `irseparated_geoflownet_intermediatel1_noreg` config, run:
```
python src/configs/save_train_config.py \
--config_filename data/configs/train_fault_deform/irseparated_geoflownet_intermediatel1_noreg.yaml \
--amp \
--model_name irseparated_GeoFlowNet \
--dataset_dir <your_directory_containing_fault_deform> \
--split_dir <your_directory_containing_the_split_files_for_fault_deform> 
```

### Pre-trained models

You can find the models trained for reproducing the paper on [huggingface](https://huggingface.co/zjuzju/microflow_models). 
You can either download the full repository, by running the following python code
```
from huggingface_hub import snapshot_download
local_dir = snapshot_download(repo_id="zjuzju/microflow_models")
```

or download each model separately using wget
```
wget https://huggingface.co/zjuzju/microflow_models/resolve/main/irseparated_geoflownet_intermediatel1/irseparated_GeoFlowNet_intermediatel1_0.8_sf012_e40.pt?download=true
```

## Inference of FaultDeform

Run the following command
```
python inference_fault_deform.py \
--train_config_name irseparated_geoflownet_intermediatel1_noreg \
--pretrained_model_filename <your_pretrained_model> \
--metric_filename <your_metric_filename> \
--save_metrics \
--dataset_name faultdeform \
--dataset_dir <FaultDeform_root_dir> \
--split_dir <your_directory_containing_the_split_files> \
--split_scaling_factors 1
```

and specify
- `pretrained_model_filename`: the path for your model (.pt)
- `metric_filename`: the path where to save your results
- `dataset_dir`: the root directory of Fault Deform
- `split_dir`: the directory containing the split files
- `split_scaling_factors`: either 0 (very small displacements), 1 (small displacements) or 2 (large displacements)

Note that your config must be located in `data/configs/train_fault_deform`.

## Inference for real examples

### Real Examples
You can evaluate our model on any pair of real-world examples, following the file structure
```
Real_example_root_dir
├── first_example_dir
│   │── <first_example_template>_pre.tif
│   └── <first_example_template>_post.tif
├── second_example_dir
│   │── <second_example_template>_pre.tif
│   └── <second_example_template>_post.tif
└── ...
```

### Inference

```
python inference_real_examples.py \
--config_filename <your_filename> \
--pretrained_model_filename <your_pretrained_model> \
--dataset_dir <first_example_dir> \
--save_dir <your_saving_directory> \
--window_size 256 \
--window_overlap 64
```

and specify 
- `pretrained_model_filename`: the path for your model (.pt)
- `dataset_dir`: the directory of your current example (for example, Real_examples/first_example)
- `save_dir`: directory where to save your estimates
- `window_size`: sliding window size, recommended for 1024
- `window_overlap`: overlap for the sliding window, (leading to a stride of window_size - 2 * window_overlap)

Note that your config must be located in `data/configs/inference_real_examples`.


## Citation
Coming soon


