import os
import torch
from src.dataloaders.datasets.fault_deform_dataset import FaultDeformDataset
from src.dataloaders.datasets.ridgecrest_dataset import  RidgeCrestDataset
from src.dataloaders.datasets.real_example_dataset import RealExampleDataset
from src.dataloaders.datasets.real_example_large_dataset import RealExampleLargeDataset
from src.dataloaders.transform import get_inference_transforms, get_train_transforms


def get_inference_dataloader(args):
    """
    Create and return the inference dataloader with appropriate transforms and dataset.
    """
    inference_transform = get_inference_transforms(args.image_size)

    if args.dataset_name.lower() == "faultdeform":
        frame_ids = load_frame_ids(args, split_name=args.split_name, split_start_idx=args.split_start_idx, split_count=args.split_count)
        inference_frame_ids = [int(frame_id) for frame_id in frame_ids]
        inference_set = FaultDeformDataset(
            frame_ids=inference_frame_ids,
            root_dir=os.path.join(args.dataset_dir, args.split_name),
            transform=inference_transform,
            scaling_factors=args.split_scaling_factors,
            fault_boundary_dir=args.fault_boundary_dir,
            fault_boundary_disk=args.fault_boundary_disk
        )
        crop_names, crs_meta_datas, transform_meta_datas = None, None, None
    elif args.dataset_name.lower() == "ridgecrest":
        crop_names = [f for f in os.listdir(args.dataset_dir) if f.startswith("corr_ads80_01") or f.startswith("corr_landsat8_01")]
        inference_set = RidgeCrestDataset(
            args.dataset_dir,
            transform=inference_transform,
            ew_template="ew_pxl.tif",  # it corresponds to the COSI-CoRR baseline in the EW direction
            ns_template="ns_pxl.tif",  # it corresponds to the COSI-CoRR baseline in the NS direction
            pre_template="pre.tif",
            post_template="post.tif",
            crop_names=crop_names
        )
        crs_meta_datas, transform_meta_datas = inference_set.get_tiff_metadata()
    elif args.dataset_name.lower() == "real_examples":
        crop_names = [f for f in os.listdir(args.dataset_dir) if f.startswith("corr_landsat8_01")]
        # crop_names = [f for f in os.listdir(args.dataset_dir) if f.startswith("corr_ads80_01") or f.startswith("corr_landsat8_01")]
        inference_set = RealExampleDataset(
            args.dataset_dir,
            transform=inference_transform,
            pre_template="pre.tif",
            post_template="post.tif",
            crop_names=crop_names
        )
        # the metadata are loaded outside of the dataloader because they are at a rasterio format, 
        # not compatible with tensors, numpy arrays, numbers, dicts or lists
        crs_meta_datas, transform_meta_datas = inference_set.get_tiff_metadata()  
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

    inference_loader = torch.utils.data.DataLoader(
        dataset=inference_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        drop_last=False)

    return inference_loader, crop_names, crs_meta_datas, transform_meta_datas


def get_real_examples_dataloader(args):
    """
    Create and return the inference dataloader with appropriate transforms and dataset.
    """
    inference_transform = get_inference_transforms(args.window_size)

    inference_set = RealExampleLargeDataset(
        args.dataset_dir,
        transform=inference_transform,
        pre_template="pre.tif",
        post_template="post.tif",
        top=0, 
        left=0, 
        window_size=args.window_size, 
        window_overlap=args.window_overlap
    )
    # the metadata are loaded outside of the dataloader because they are at a rasterio format, 
    # not compatible with tensors, numpy arrays, numbers, dicts or lists
    crs_meta_datas, transform_meta_datas = inference_set.get_tiff_metadata()  

    inference_loader = torch.utils.data.DataLoader(
        dataset=inference_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        drop_last=False)

    return inference_loader, crs_meta_datas, transform_meta_datas


def get_training_dataloaders(args):
    train_frame_ids = load_frame_ids(args, split_name="train", split_start_idx=0, split_count=args.train_count)
    train_transform = get_train_transforms(args.train_image_size)
    train_set = FaultDeformDataset(
        frame_ids=train_frame_ids,
        root_dir=os.path.join(args.dataset_dir, "train"),
        transform=train_transform,
        scaling_factors=args.train_scaling_factors,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        drop_last=args.drop_last)

    val_frame_ids = load_frame_ids(args, split_name="val", split_start_idx=0, split_count=args.val_count)
    validation_transform = get_inference_transforms(args.train_image_size)
    val_set = FaultDeformDataset(
        frame_ids=val_frame_ids,
        root_dir=os.path.join(args.dataset_dir, "validation"),
        transform=validation_transform,
        scaling_factors=args.val_scaling_factors,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        drop_last=args.drop_last)
    print('{} train samples and {} validation samples'.format(len(train_set), len(val_set)))
    return train_loader, val_loader


def load_frame_ids(args, split_name="test", split_start_idx=0, split_count=-1):
    """
    Load frame IDs corresponding to the dataset split file.
    """
    split_filename = os.path.join(args.split_dir, f"{split_name}_names_center.txt" if ("near_fault_only" in args and args.near_fault_only) else f"{split_name}_names.txt")
    with open(split_filename, "r") as reader:
        split_names = [int(name.strip().split("_")[0]) for name in reader]
   
    if split_count != -1:
        frame_ids = split_names[split_start_idx:split_start_idx + split_count]

    return frame_ids


