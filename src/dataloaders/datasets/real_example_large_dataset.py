import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import os
from src.dataloaders.tiff_utils import open_tiff, get_tiff_band, get_tiff_crs, get_tiff_transform


class RealExampleLargeDataset(Dataset):

    def __init__(
        self, root_dir, transform=None, pre_template="pre.tif", post_template="post.tif", top=0, left=0, window_size=1024, window_overlap=64
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

        self.load_patches_in_memory()

    def load_patches_in_memory(self):
        pre_filename, post_filename = self.get_filenames(self.root_dir)

        pre_post = self.load_prepost_tensor(pre_filename, post_filename)

        # Extract crs metadata
        pre_img = open_tiff(pre_filename)
        self.crs_info = get_tiff_crs(pre_img)
        self.transform_info = get_tiff_transform(pre_img)

        # Set image size
        self.height = pre_post.shape[-2]
        self.width = pre_post.shape[-1]
        print(f"in load h={self.height} w={self.width}")

        # Extract the patches in memory
        self.pre_posts, self.x_positions, self.y_positions = self.extract_patches(pre_post)

    def get_filenames(self, example_dir):
        filenames = os.listdir(example_dir)
        pre_filename, post_filename = None, None
        for filename in filenames:
            if filename.endswith(self.pre_template):
                pre_filename = os.path.join(example_dir, filename)
                # set the image_pair_name 
                self.image_pair_name = filename.replace(self.pre_template, "")
            if filename.endswith(self.post_template):
                post_filename = os.path.join(example_dir, filename)
      
        return pre_filename, post_filename

    def load_prepost_tensor(self, pre_filename, post_filename):
        """ Loads prepost tensor """
        pre_img = open_tiff(pre_filename)
        pre_tensor = torch.tensor(get_tiff_band(pre_img, 1))
        
        post_img = open_tiff(post_filename)
        post_tensor = torch.tensor(get_tiff_band(post_img, 1))
        
        return torch.stack([pre_tensor, post_tensor])

    def __len__(self):
        return len(self.pre_posts)
    
    def get_image_info(self):
        """ Get the tiff metadata, which are not at the right format to be yield by the dataloader """
        return self.height, self.width, self.window_overlap, self.window_size, self.image_pair_name
    
    def get_tiff_metadata(self):
        """ Get the tiff metadata, which are not at the right format to be yield by the dataloader """
        return self.crs_info, self.transform_info

    def extract_patches(self, images):
        def update_patches(images, x, y, window_size, patches, x_positions, y_positions):
            patch = images[:, y: y + window_size, x: x + window_size]
            if torch.sum(torch.abs(patch)) == 0:  # for regions where there is no data
                return
            patches.append(patch)
            x_positions.append(x)
            y_positions.append(y)

        stride = self.window_size - 2 * self.window_overlap
        _, h, w = images.shape
        patches, x_positions, y_positions = [], [], []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Handling edge cases
                if y + self.window_size > h:
                    y = h - self.window_size
                if x + self.window_size > w:
                    x = w - self.window_size
                update_patches(images, x, y, self.window_size, patches, x_positions, y_positions)

        return np.array(patches), x_positions, y_positions

    def __getitem__(self, idx):
        sample = {
            'pre_post_image': torch.tensor(self.pre_posts[idx]),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['frame_id'] = torch.tensor(idx)
        sample['x_position'] = self.x_positions[idx]
        sample['y_position'] = self.y_positions[idx]

        return sample

