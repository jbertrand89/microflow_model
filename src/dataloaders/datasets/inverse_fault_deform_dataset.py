import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import os
import math
import cv2
from skimage.morphology import disk
from src.dataloaders.datasets.fault_deform_dataset import FaultDeformDataset


class InverseFaultDeformDataset(FaultDeformDataset):

    def __init__(
            self, frame_ids, root_dir, transform=None, scaling_factors=None, fault_boundary_dir=None,
            fault_boundary_disk=1,
            inverse_root_dir=None
    ):
        super().__init__(frame_ids, root_dir, transform, scaling_factors, fault_boundary_dir, fault_boundary_disk)
        self.inverse_root_dir = inverse_root_dir

    def load_inverse_sample(self, filename):
        arrays = np.load(filename)

        dm_ew = arrays["inverse_flow_ew"]
        dm_ns = arrays["inverse_flow_ns"]
        dm_array = np.stack([dm_ew, dm_ns]).astype(np.float32)

        return dm_array

    def __getitem__(self, idx):
        id_scalings = idx % self.len_scaling_factor
        id_frames = idx // self.len_scaling_factor

        frame_id = self.frame_ids[id_frames]
        scaling_factor = self.scaling_factors[id_scalings]

        sample_filename = os.path.join(self.root_dir, f"{frame_id:06d}_{scaling_factor}_sample.npy.npz")
        pre_post_image_array, dm_array = self.load_sample(sample_filename)

        inverse_filename = os.path.join(self.inverse_root_dir, f"{frame_id:06d}_{scaling_factor}_inverse.npy.npz")
        inverse_dm_array = self.load_inverse_sample(inverse_filename)

        max_displacement = np.abs(np.max(dm_array))

        target_dm = torch.from_numpy(inverse_dm_array)
        sample = {
            'pre_post_image': torch.from_numpy(pre_post_image_array),
            'target_dm': target_dm,
            'valid': torch.ones_like(target_dm, dtype=torch.bool)
        }

        if self.fault_boundary_dir is not None:
            fault_filename = os.path.join(self.fault_boundary_dir, f"{frame_id:06d}_fault_boundary.npy")
            fault_boundary = np.load(fault_filename)  # values are either 0 or 255
            fault_boundary = cv2.dilate(fault_boundary.astype(np.uint8),
                                        disk(self.fault_boundary_disk).astype(np.uint8))
            sample['fault_boundary'] = torch.from_numpy(fault_boundary).unsqueeze(dim=0) / 255

        if self.transform is not None:
            sample = self.transform(sample)

        sample['frame_id'] = torch.tensor(frame_id)
        sample['scaling_factor'] = torch.tensor(int(scaling_factor))
        sample['max_displacement'] = torch.tensor(int(math.ceil(max_displacement)))

        return sample
