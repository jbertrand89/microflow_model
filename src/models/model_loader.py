import torch
from collections import OrderedDict
from src.models.iterative_models.iterative_separated_weights import get_iterative_separated_weights_model
from src.models.iterative_models.iterative_shared_weights import get_iterative_shared_weights_model
from src.models.baselines.raft.raft import RAFT


def load_model(args, device):
    """
    Load the model for the configuration.
    """
    if args.model_name.lower().startswith("raft"):
        model = load_raft_model(args, device)
    elif args.model_name.lower().startswith("ir"):
        model = load_iterative_explicit_warping_model(args, device)
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    return model


def load_raft_model(args, device):

    model = RAFT(args)

    if args.pretrained_model_filename != "":
        checkpoint = torch.load(args.pretrained_model_filename)
        new_state_dict = OrderedDict()
        for key, value in checkpoint['model_state_dict'].items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)

    model.to(device)

    return model


def load_iterative_explicit_warping_model(args, device):
    pretrained_end2end = args.pretrained_model_filename != ""
    filenames = [args.pretrained_model_filename] if pretrained_end2end else []

    iterative_name, backbone_name = args.model_name.lower().split("_")

    if iterative_name == "irshared":
        model = get_iterative_shared_weights_model(
            device, backbone_name, False, args.max_iterations, filenames, trained_end2end=pretrained_end2end)
    elif iterative_name == "irseparated":
        model = get_iterative_separated_weights_model(
            device, backbone_name, False, args.max_iterations, filenames, args=args,
            trained_end2end=pretrained_end2end)
    else:
        raise ValueError(f"Model {iterative_name} not supported")

    return model
