import argparse
import os
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--config_filename', type=str)

    # model parameters
    parser.add_argument('--model_name', default='irseparated_GeoFlowNet', help='network')
    parser.add_argument('--max_iterations', default=3, type=int, help='Image size')
    parser.add_argument('--repeat_model', action='store_true', help='')
    parser.add_argument('--repeat_first_iteration', action='store_true', help='')
    parser.add_argument('--use_batch_norm', action='store_true', help='use batch normalization')

    # regularization parameters
    parser.add_argument('--regularization', action='store_true', help='use the regularization')
    parser.add_argument('--penalty_function', type=str, default = "ltv", choices=["ltv", "tv", "l2"], help='which penalty function to use')
    parser.add_argument('--reg_lambda', default=0.001, type=float, help='lambda parameter for ltv')
    parser.add_argument('--reg_iterations', default=3, type=int, help='k parameter for ltv') 
    parser.add_argument('--reg_2d_max_iter', default=10, type=int, 
                        help='number of 2D iterations (alternating between rows and columns)') 
    parser.add_argument('--reg_threads', default=16, type=int, help='number of cpu threads') 

    # raft parameters
    parser.add_argument('--raft_small', action='store_true', help='use the crop resize data augmentation')
    parser.add_argument('--raft_input_channel', default=3, type=int, help='max factor for resize')
    parser.add_argument('--raft_dropout', type=float, default=0.0)

    # loss and metrics parameters:
    parser.add_argument('--loss', default='raft', help='Loss used')
    parser.add_argument('--gamma', type=float, default=0.8, help='gamma for intermediate l1 loss')

    # training parameters
    parser.add_argument('--weight_decay', '--wd', default=4e-4, type=float, help='weight decay')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--momemtum', default=0.9, type=float, help='momentum')
    parser.add_argument('--beta', default=0.999, type=float, help='beta parameter for adam')  
    parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'],  help='solver algorithms')
    parser.add_argument('--scheduler_name', type=str, default="multi_steps_lr")
    parser.add_argument('--epochs', default=40, type=int, help='number of total epochs to run') 
    parser.add_argument('--gradient_clip', default=1., type=float, help='gradient clipping value')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size')

    # dataset parameters
    parser.add_argument('--dataset_name', default='faultdeform', help='dataset name')
    parser.add_argument('--train_count', type=int, default=20000, help='number of train examples')
    parser.add_argument('--train_scaling_factors', nargs='+', default=[0, 1, 2], help="train scaling factors")
    parser.add_argument('--val_count', type=int, default=5000, help='number of val examples')
    parser.add_argument('--val_scaling_factors',  nargs='+', default=[0, 1, 2], help="val scaling factors")   
    parser.add_argument('--train_image_size', default=256, type=int, help='Image size')

    # system parameters
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--num_workers', default=16, type=int, help='num workers in dataloader')
    parser.add_argument('--persistent_workers', default=True, action=argparse.BooleanOptionalAction,
                        help='activate persistent workers in dataloader')
    parser.add_argument('--pin_memory', default=True, action=argparse.BooleanOptionalAction,
                        help='activate pin memory option in dataloader')
    parser.add_argument('--non_blocking', default=True, action=argparse.BooleanOptionalAction,
                        help='activate asynchronuous GPU transfer')
    parser.add_argument('--prefetch_factor', default=3, type=int,
                        help='prefectch factor in dataloader')
    parser.add_argument('--drop_last', action='store_true', help='')

    # add parser arguments
    args = parser.parse_args()

    return args


def save_config(config_filename, args):
    iterative_model_parameters = {
        "MODEL_NAME": args.model_name,
        "MAX_ITERATIONS": args.max_iterations,
        "REPEAT_MODEL": args.repeat_model,
        "REPEAT_FIRST_ITERATION": args.repeat_first_iteration,
        "USE_BATCH_NORM": args.use_batch_norm,
        "LOSS": args.loss,
        "GAMMA": args.gamma
    }

    regularization_parameters = {
        "REGULARIZATION": args.regularization,
        "PENALTY_FUNCTION": args.penalty_function,
        "REG_LAMBDA": args.reg_lambda,
        "REG_ITERATIONS": args.reg_iterations,
        "REG_2D_MAX_ITER": args.reg_2d_max_iter,
        "REG_THREADS": args.reg_threads
    }

    baseline_parameters = {
        "RAFT_SMALL": args.raft_small,
        "RAFT_INPUT_CHANNEL": args.raft_input_channel,
        "RAFT_DROPOUT": args.raft_dropout,
    }  

    training_parameters = {
        "SOLVER": args.solver,
        "LEARNING_RATE": args.learning_rate,
        "WEIGHT_DECAY": args.weight_decay,
        "MOMENTUM": args.momemtum,
        "BETA": args.beta,
        "EPOCHS": args.epochs,
        "TRAIN_BATCH_SIZE": args.train_batch_size,
        "GRADIENT_CLIP": args.gradient_clip,
        "SCHEDULER_NAME": args.scheduler_name,
    }

    train_dataset_parameters = {
        "DATASET_NAME": args.dataset_name,
        "TRAIN_IMAGE_SIZE": args.train_image_size,
        "TRAIN_COUNT": args.train_count,
        "VAL_COUNT": args.val_count,
        "TRAIN_SCALING_FACTORS": args.train_scaling_factors,
        "VAL_SCALING_FACTORS": args.val_scaling_factors,
    }

    system_parameters = {
        "AMP": args.amp,
        "NUM_WORKERS": args.num_workers,
        "PERSISTENT_WORKERS": args.persistent_workers,
        "PIN_MEMORY": args.pin_memory,
        "NON_BLOCKING": args.non_blocking,
        "PREFETCH_FACTOR": args.prefetch_factor,
        "DROP_LAST": args.drop_last
    }

    parameters = {
        "SYSTEM": system_parameters,
        "ITERATIVE_MODEL": iterative_model_parameters,
        "REGULARIZATION": regularization_parameters,
        "BASELINE_MODELS": baseline_parameters,
        "TRAINING": training_parameters,
        "TRAIN_DATASET": train_dataset_parameters,
    }

    with open(config_filename, 'w') as file:
        yaml.dump(parameters, file)


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.config_filename), exist_ok=True)
    save_config(args.config_filename, args)
    print(f"saved config {args.config_filename}")


