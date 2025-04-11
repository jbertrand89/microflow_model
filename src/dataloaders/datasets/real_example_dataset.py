import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import os

from src.dataloaders.tiff_utils import open_tiff, get_tiff_band, get_tiff_crs, get_tiff_transform


class RealExampleDataset(Dataset):

    def __init__(
            self, root_dir, transform=None, pre_template="pre.tif", post_template="post.tif", crop_names=None,
    ):
        self.root_dir = root_dir  # contains all the input-pair folders
        self.transform = transform
        self.pre_template = pre_template
        self.post_template = post_template
        self.crop_names = crop_names

        # Tiff specific data, for compatibility with QGis
        self.crs_info = [None] * len(self.crop_names)
        self.transform_info = [None] * len(self.crop_names)
        self.load_tiff_metadata()

    def get_filenames(self, example_dir):
        filenames = os.listdir(example_dir)
        pre_filename, post_filename = None, None
        for filename in filenames:
            if filename.endswith(self.pre_template):
                pre_filename = os.path.join(example_dir, filename)
            if filename.endswith(self.post_template):
                post_filename = os.path.join(example_dir, filename)
        return pre_filename, post_filename
    
    def load_tiff_metadata(self):
        "Load tiff metadata to align the output images in QGis"
        for idx, crop_name in enumerate(self.crop_names):
            example_dir = os.path.join(self.root_dir, crop_name)
            pre_filename, _ = self.get_filenames(example_dir)

            pre_img = open_tiff(pre_filename)
            self.crs_info[idx] = get_tiff_crs(pre_img)
            self.transform_info[idx] = get_tiff_transform(pre_img)

    def get_crop_names(self):
        return self.crop_names

    def get_tiff_metadata(self):
        """ Get the tiff metadata, which are not at the right format to be yield by the dataloader """
        return self.crs_info, self.transform_info

    def __len__(self):
        return len(self.crop_names)
    
    def load_prepost_tensor(self, pre_filename, post_filename):
        """ Loads prepost tensor """
        pre_img = open_tiff(pre_filename)
        pre_tensor = torch.tensor(get_tiff_band(pre_img, 1))
        
        post_img = open_tiff(post_filename)
        post_tensor = torch.tensor(get_tiff_band(post_img, 1))
        
        return torch.stack([pre_tensor, post_tensor])

    def __getitem__(self, idx):
        crop_name = self.crop_names[idx]
        example_dir = os.path.join(self.root_dir, crop_name)
        pre_filename, post_filename = self.get_filenames(example_dir)

        pre_post_tensor = self.load_prepost_tensor(pre_filename, post_filename)
        print(f"pre_post_tensor {pre_post_tensor.shape}")
        raise Exception("juju")
        
        sample = {
            'pre_post_image': pre_post_tensor,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['frame_id'] = torch.tensor(idx)

        return sample


