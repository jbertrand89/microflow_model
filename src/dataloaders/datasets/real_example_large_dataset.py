import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import os
from src.dataloaders.tiff_utils import open_tiff, get_tiff_band, get_tiff_crs, get_tiff_transform
import torch
import rasterio as rio
from rasterio.windows import Window
import math


class RealExampleLargeDataset(Dataset):

    def __init__(
            self, root_dir, transform=None, pre_template="pre.tif", post_template="post.tif", top=0, left=0,
            window_size=1024, window_overlap=64
    ):
        self.root_dir = root_dir  # contains all the input-pair folders
        self.transform = transform
        self.pre_template = pre_template
        self.post_template = post_template
        self.top = top
        self.left = left
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.height = -1
        self.width = -1
        self.image_pair_name = None
        self.pre_filename, self.post_filename = self.get_filenames(self.root_dir)

        # load image info from rasterio (efficiently implemented)
        with rio.open(self.pre_filename) as pre_raster:
            self.width = pre_raster.width
            self.height = pre_raster.height
            self.crs_info = pre_raster.crs
            self.transform_info = pre_raster.transform

    def get_filenames(self, example_dir):
        filenames = os.listdir(example_dir)
        pre_filename, post_filename = None, None
        for filename in filenames:
            if filename.endswith(self.pre_template):
                pre_filename = os.path.join(example_dir, filename)
                self.image_pair_name = filename.replace(self.pre_template, "")
            if filename.endswith(self.post_template):
                post_filename = os.path.join(example_dir, filename)
      
        return pre_filename, post_filename

    def __len__(self):
        self.height_correlation = math.ceil((self.height - self.window_size) / self.window_overlap) + 1
        self.width_correlation = math.ceil((self.width - self.window_size) / self.window_overlap) + 1
        return self.height_correlation * self.width_correlation

    def get_image_info(self):
        """ Get the tiff metadata, which are not at the right format to be yield by the dataloader """
        return self.height, self.width, self.window_overlap, self.window_size, self.image_pair_name
    
    def get_tiff_metadata(self):
        """ Get the tiff metadata, which are not at the right format to be yield by the dataloader """
        return self.crs_info, self.transform_info

    def __getitem__(self, idx):
        top = (idx // self.width_correlation) * self.window_overlap
        left = (idx % self.width_correlation) * self.window_overlap

        if top + self.window_size > self.height:
            top = self.height - self.window_size
        if left + self.window_size > self.width:
            left = self.width - self.window_size
        window = Window(
            col_off=left, row_off=top, width=self.window_size, height=self.window_size)

        with rio.open(self.pre_filename) as pre_raster:
            window_pre = pre_raster.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)

        with rio.open(self.post_filename) as post_raster:
            window_post = post_raster.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)

        # Convert to tensors
        window_pre = torch.from_numpy(window_pre)
        window_post = torch.from_numpy(window_post)

        sample = {
            'pre_post_image': torch.stack([window_pre, window_post]),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['frame_id'] = torch.tensor(idx)
        sample['x_position'] = left
        sample['y_position'] = top

        return sample
