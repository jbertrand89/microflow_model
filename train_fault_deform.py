import os
import argparse
import torch
import numpy as np
import random

from src.dataloaders.data_loader import get_training_dataloaders
from src.models.model_loader import load_model
from src.metrics_and_losses.photometric_metrics import realEPE
from src.metrics_and_losses.losses import IntermediateL1Loss
from src.training_inference.train import train
from src.configs.load_config import ConfigArgumentParser, YamlConfigAction

rng = 1


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """
    Parse command line arguments for the inference script.
    """
    parser = argparse.ArgumentParser()

    parser = ConfigArgumentParser('training on fault deform')
    parser.add_argument('--train_config_name', default='data/configs/train/microflow.yaml')
    parser.add_argument('--seed', default=1, type=int, help='Seed used for the random number generators')
    
    # directories parameters
    parser.add_argument('--split_dir', type=str, help='directory containing the dataset splits')
    parser.add_argument('--dataset_dir', help='path to dataset dir')
    parser.add_argument('--save_offline', action='store_true', help='If true, save offline logs to be used for wandb')
    parser.add_argument('--offline_dir', type=str, default="", help='Directory where to save the offline logs')
    parser.add_argument('--checkpoints_dir', default='/gpfswork/rech/nrl/ubt63as/subpixel_sat_ml/results/',
                        help='Directory where to save the checkpoints')
    
    # pretraining parameters
    parser.add_argument('--training_from_pretrain', action='store_true',
        help='if true, restart the training parameters from the pretrained model. Else start from scratch')
    parser.add_argument('--restart', action='store_true', help='if True, restart the training from where it stopped')
    parser.add_argument('--pretrained_model_filename', type=str, default="", help="Path to pretrained model. If restart is True, it is overwritten")

    namespace, _ = parser.parse_known_args()

    config_filename = os.path.join("data/configs/train_fault_deform/", f'{namespace.train_config_name}.yaml')
    parser.add_argument('--config', action=YamlConfigAction, default=[config_filename])
    args = parser.parse_args()

    # Create checkpoints directory, 
    args.save_path = os.path.join(args.checkpoints_dir, "checkpoints")
    os.makedirs(args.save_path, exist_ok=True)

    # Create offline directory
    if args.save_offline:
        os.makedirs(args.offline_dir, exist_ok=True)
        args.save_offline_filename = os.path.join(args.offline_dir, "offline_logs.txt")

    if args.restart:
        args.training_from_pretrain = True
        args.pretrained_model_filename = os.path.join(args.save_path, "checkpoint_last_model.pt")
    else:
        if args.save_offline:
            save_epoch = [
                "epoch","train_loss", "val_loss", "train_mae",
                "val_mae", "train_epe", "val_epe", "lr", "scaling_factor_errors", "train_step_errors", "val_step_errors"]
            with open(args.save_offline_filename, "w+") as writer:
                writer.write(";".join(save_epoch) + "\n")

        print('\nInput Arguments')
        print('---------------')
        for k, v in sorted(dict(vars(args)).items()):
            print('%s: %s' % (k, str(v)))
        print()

    return args


def init_optimizer(args, model):
    if args.solver == "AdamW": # used in RAFT paper
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.wdecay, eps=args.epsilon)
    elif args.solver == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    return optimizer


def get_loss_functions(args):
    if args.loss.lower() == "mse":
        loss_function = torch.nn.MSELoss()
    elif args.loss.lower() == "l1":
        loss_function = torch.nn.L1Loss()
    elif args.loss.lower() == "raft":
        loss_function = IntermediateL1Loss(args.gamma)
    elif args.loss.lower() == "intermediate_l1":
        loss_function = IntermediateL1Loss(args.gamma)
    else:
        raise Exception("Loss not implemented")
    return loss_function


def run_train(args):
    # Fix the seed
    fix_seed(args.seed)

    # Define device
    device = torch.device('cuda')

    # Load model
    model = load_model(args, device)

    # Get dataloaders
    train_loader, val_loader = get_training_dataloaders(args)
    train_dataset_count = len(train_loader)
    val_dataset_count = len(val_loader)
    print(f"train examples={train_dataset_count} val examples={val_dataset_count}")

    # Loss and metrics
    regression_loss_function = get_loss_functions(args)
    epe = realEPE
    mae = torch.nn.L1Loss()
    metrics = [epe, mae]

    # Optimizer
    optimizer = init_optimizer(args, model)

    # Scheduler
    if args.scheduler_name.lower() == "one_cycle_lr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, args.lr, args.num_steps + 100, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    elif args.scheduler_name.lower() == "multi_steps_lr":
        milestones = [20, 40, 80]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5) 
    else:
        scheduler = None
    print(f"scheduler {scheduler}")

    # Restore epoch, scheduler, and optimizer
    start_epoch = 0
    if args.pretrained_model_filename and args.training_from_pretrain:
        checkpoint = torch.load(args.pretrained_model_filename)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    train(args, device, model, train_loader, val_loader, optimizer, scheduler, start_epoch, regression_loss_function, metrics)


if __name__ == '__main__':
    # Load parameters
    args = parse_args()

    run_train(args)


