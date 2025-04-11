import argparse
import torch
import numpy as np
import os
import time
import random

from src.models.model_loader import load_model
from src.training_inference.evaluate import evaluate_inference_large_image
from src.dataloaders.data_loader import get_real_examples_dataloader
from src.configs.load_config import ConfigArgumentParser, YamlConfigAction


def fix_seed(seed):
    """
    Fix the random seed for reproducibility across PyTorch, NumPy, and Python's random module.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """
    Parse command line arguments for estimating the displacement fields for large images.
    """
    parser = ConfigArgumentParser('inference real examples')
    parser.add_argument('--config_name', default='microflow')
    
    # Inference specific parameters
    parser.add_argument('--seed', default=1, type=int, help='Seed used for the random number generators')
    parser.add_argument('--pretrained_model_filename', help='Filename containing the pretrained weights associated witht the config')  
    parser.add_argument('--dataset_dir', help='path to the dir containing the real examples')  
    parser.add_argument('--save_dir', default=None, help="dir where to save the flow estimates")
    
    parser.add_argument('--window_size', default=1024, type=int, help='Sliding window size (useful when the image is large)')  
    parser.add_argument('--window_overlap', default=0, type=int, help='Overlap between the sliding windows (stride of window_size - 2*window_overlap)')  
    parser.add_argument('-b', '--batch-size', default=1, type=int, help='batch size for estimating on the sliding windows')  
    namespace, _ = parser.parse_known_args()

    config_filename = os.path.join("data/configs/inference_real_examples/", f'{namespace.config_name}.yaml')
    parser.add_argument('--config', action=YamlConfigAction, default=[config_filename])

    return parser.parse_args()


def run_inference(args):
    """
    Run the inference pipeline.
    """
    # Fix the seed
    fix_seed(args.seed)

    # Define device
    device = torch.device('cuda')

    # Load models
    optical_flow_model = load_model(args, device)

    # Get dataloaders
    loader, crs_meta_datas, transform_meta_datas = get_real_examples_dataloader(args)

    t_start = time.time()   
    evaluate_inference_large_image(
        args=args,
        device=device,
        model=optical_flow_model,
        loader=loader,  
        crs_meta_datas=crs_meta_datas,
        transform_meta_datas=transform_meta_datas,
        )
    t_end = time.time()

    print(f"total time {t_end - t_start}")


if __name__ == '__main__':

    # Load parameters
    args = parse_args()

    run_inference(args)
