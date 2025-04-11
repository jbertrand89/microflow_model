import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
import cv2
from skimage.morphology import disk


class FaultDeformDataset(Dataset):

    def __init__(
            self, frame_ids, root_dir, transform=None, scaling_factors=None, fault_boundary_dir=None, fault_boundary_disk=1
    ):
        self.frame_ids = frame_ids
        self.root_dir = root_dir  # contains the .npy.npz files
        self.transform = transform
        self.scaling_factors = scaling_factors
        self.len_scaling_factor = len(self.scaling_factors)

        self.fault_boundary_dir = fault_boundary_dir
        self.fault_boundary_disk = fault_boundary_disk

    def __len__(self):
        return len(self.frame_ids) * self.len_scaling_factor

    def load_sample(self, filename):
        arrays = np.load(filename)

        pre = arrays["int_array_1"]
        post = arrays["int_array_2"]
        pre_post_image_array = np.stack([pre, post]).astype(np.float32)

        dm_ew = arrays["float_array_1"]
        dm_ns = arrays["float_array_2"]
        dm_array = np.stack([dm_ew, dm_ns]).astype(np.float32)

        return pre_post_image_array, dm_array

    def __getitem__(self, idx):
        id_scalings = idx % self.len_scaling_factor
        id_frames = idx // self.len_scaling_factor

        frame_id = self.frame_ids[id_frames]
        scaling_factor = self.scaling_factors[id_scalings]
        sample_filename = os.path.join(self.root_dir, f"{frame_id:06d}_{scaling_factor}_sample.npy.npz")

        pre_post_image_array, dm_array = self.load_sample(sample_filename)
        max_displacement = np.abs(np.max(dm_array))

        if self.fault_boundary_dir is not None:
            fault_filename = os.path.join(self.fault_boundary_dir, f"{frame_id:06d}_fault_boundary.npy")
            fault_boundary = np.load(fault_filename)  # values are either 0 or 255
            fault_boundary = cv2.dilate(fault_boundary.astype(np.uint8), disk(self.fault_boundary_disk).astype(np.uint8))

            sample = {
                'pre_post_image': torch.from_numpy(pre_post_image_array),
                'target_dm': torch.from_numpy(dm_array),
                'fault_boundary': torch.from_numpy(fault_boundary).unsqueeze(dim=0) / 255
            }

        else:
            sample = {
                'pre_post_image': torch.from_numpy(pre_post_image_array),
                'target_dm': torch.from_numpy(dm_array)
            }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['frame_id'] = torch.tensor(frame_id)
        sample['scaling_factor'] = torch.tensor(int(scaling_factor))
        sample['max_displacement'] = torch.tensor(int(math.ceil(max_displacement)))

        return sample
