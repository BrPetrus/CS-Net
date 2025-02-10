import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import warnings
import numpy as np
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from numpy.typing import NDArray
from typing import Tuple, Optional
from pathlib import Path
import skimage.io as skio

class Data(Dataset):
    def __init__(self,
                 root_dir: str,
                 train=True,
                 scale=512):
        self.root_dir = root_dir
        self.train = train
        self.scale = scale
        self.images = []
        self.groundtruth = []
        self._load_dataset()
    
    def _load_stack(self, dir_path: str) -> NDArray:
        stack = []
        # for file in glob.glob(str(Path(dir_path) / '*.jpg')):
        #     stack.append(skio.imread(file))
        for file in sorted(os.listdir(dir_path)):
            stack.append(skio.imread(str(Path(dir_path) / file )))

        # np_stack = np.stack(stack, axis=0).astype(np.float32)
        # np_stack /= 255.0 
        # np_stack = np_stack.transpose(1, 2, 0)  # HWC
        return np.array(stack)
    
    def _random_crop(self, image: NDArray, label: NDArray, crop_size) -> Tuple[NDArray, NDArray]:
        w, h, d = image.shape
        z = random.randint(0, w - crop_size[0])
        y = random.randint(0, h - crop_size[1])
        x = random.randint(0, d - crop_size[2])

        image = image[z:z + crop_size[0], y:y + crop_size[1], x:x + crop_size[2]]
        label = label[z:z + crop_size[0], y:y + crop_size[1], x:x + crop_size[2]]
        return image, label

    def _load_dataset(self):
        root_path = Path(self.root_dir)
        images = []
        groundtruth = []

        # Setup paths
        if self.train:
            data_path = root_path / 'training'
        else:
            data_path = root_path / 'test'

        for directory in os.listdir(str(data_path)):
            input_image_path = data_path / directory / 'noise_image_0'
            groundtruth_path = data_path / directory / 'original_image'

            if not input_image_path.exists() or not groundtruth_path.exists():
                print(f"Skipping {input_image_path} and {groundtruth_path}")

            # Load the images
            img = self._load_stack(str(input_image_path)).astype(np.float32)
            gt_raw = self._load_stack(str(groundtruth_path)).astype(np.int64)
            gt = np.zeros_like(gt_raw)
            gt[gt_raw > 128] = 255
            

            # Transpose
            img = img.transpose(2, 0, 1)  # [x, y, z] -> [z, x, y]
            gt = gt.transpose(2, 0, 1)

            # Cut to replicate MRA Brain Loader
            # TODO stufy more
            # img = img[:64, :104, :112]
            # gt = gt[:64, :104, :112]
            # img = img[:100, :100, :100]
            # gt = gt[:100, :100, :100]
            # img = img[:64, :64, :64]
            # gt = gt[:64, :64, :64]


            # # Expand dimensions
            # img = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)
            # gt = torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0)

            # # Normalize
            # img = img / 255.0
            # gt = gt // 255

            images.append(img)
            groundtruth.append(gt)

        # self.images = np.array(images)
        # self.groundtruth = np.array(groundtruth)
        self.images = images
        self.groundtruth = groundtruth
        print(f"Loaded {len(images)} images and {len(groundtruth)} groundtruth images")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx) -> Optional[Tuple[NDArray[np.float32], NDArray[np.int64]]]:
        if idx >= len(self.images):
            return None
        data, label = self.images[idx], self.groundtruth[idx]

        # Apply the random crop
        data, label = self._random_crop(data, label, (64, 64, 64))

        # Expand dimensions
        data = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
        label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)

        # Normalize
        data /= 255.0
        label //= 255
        return data, label
        


