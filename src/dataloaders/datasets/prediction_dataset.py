import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
from skimage.morphology import disk
from src.dataloaders.tiff_utils import open_tiff, get_tiff_band


class PredictionDataset(Dataset):

    def __init__(
            self, frame_ids, fault_deform_root_dir, estimation_root_dir, 
            transform=None, scaling_factors=None, fault_boundary_dir=None, fault_boundary_disk=1
    ):
        self.frame_ids = frame_ids
        self.fault_deform_root_dir = fault_deform_root_dir  # contains the .npy.npz files
        self.estimation_root_dir = estimation_root_dir
        self.transform = transform
        self.scaling_factors = scaling_factors
        self.len_scaling_factor = len(self.scaling_factors)

        self.fault_boundary_dir = fault_boundary_dir
        self.fault_boundary_disk = fault_boundary_disk

    def __len__(self):
        return len(self.frame_ids) * self.len_scaling_factor

    def load_gt_dm(self, filename):
        arrays = np.load(filename)

        dm_ew = arrays["float_array_1"]
        dm_ns = arrays["float_array_2"]
        dm_array = np.stack([dm_ew, dm_ns]).astype(np.float32)

        return dm_array

    def load_estimated_dm(self, ew_filename, ns_filename):
        """ Load estimated dm """
        ew_img = open_tiff(ew_filename)
        ew_tensor = np.array(get_tiff_band(ew_img, 1))
        
        ns_img = open_tiff(ns_filename)
        ns_tensor = np.array(get_tiff_band(ns_img, 1))
        
        return np.array([ew_tensor, ns_tensor])

    def __getitem__(self, idx):
        id_scalings = idx % self.len_scaling_factor
        id_frames = idx // self.len_scaling_factor

        frame_id = self.frame_ids[id_frames]
        scaling_factor = self.scaling_factors[id_scalings]

        sample_filename = os.path.join(self.fault_deform_root_dir, f"{frame_id:06d}_{scaling_factor}_sample.npy.npz")
        gt_dm = self.load_gt_dm(sample_filename)

        ew_filename = os.path.join(self.estimation_root_dir, f"{frame_id:06d}_{scaling_factor}_ew.tif")
        ns_filename = os.path.join(self.estimation_root_dir, f"{frame_id:06d}_{scaling_factor}_ns.tif")
        estimated_dm = self.load_estimated_dm(ew_filename, ns_filename)

        if self.fault_boundary_dir is not None:
            fault_filename = os.path.join(self.fault_boundary_dir, f"{frame_id:06d}_fault_boundary.npy")
            fault_boundary = np.load(fault_filename)  # values are either 0 or 255
            fault_boundary = cv2.dilate(fault_boundary.astype(np.uint8), disk(self.fault_boundary_disk).astype(np.uint8))

            sample = {
                # 'estimated_dm': estimated_dm,
                'estimated_dm': torch.from_numpy(estimated_dm),
                'target_dm': torch.from_numpy(gt_dm),
                'fault_boundary': torch.from_numpy(fault_boundary).unsqueeze(0) / 255
            }

        else:
            sample = {
                # 'estimated_dm': torch.from_numpy(estimated_dm).unsqueeze(0),
                'estimated_dm': torch.from_numpy(estimated_dm),
                'target_dm': torch.from_numpy(gt_dm)
            }

        if self.transform is not None:
            sample = self.transform(sample)

        # print(f"sample estimated_dm {sample['estimated_dm'].shape}")
        # print(f"sample target_dm {sample['target_dm'].shape}")
        # print(f"sample fault_boundary {sample['fault_boundary'].shape}")
        # raise Exception("j")

        sample['frame_id'] = torch.tensor(frame_id)
        sample['scaling_factor'] = torch.tensor(int(scaling_factor))

        return sample
