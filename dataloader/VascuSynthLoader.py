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
                 train=True):
        self.root_dir = root_dir
        self.train = train
        self.images = []
        self.groundtruth = []
        self._load_dataset()
    
    def _load_stack(self, dir_path: str) -> NDArray:
        stack = []
        for file in sorted(os.listdir(dir_path)):
            stack.append(skio.imread(str(Path(dir_path) / file )))

        # np_stack = np.stack(stack, axis=0).astype(np.float32)
        # np_stack /= 255.0 
        # np_stack = np_stack.transpose(1, 2, 0)  # HWC
        return np.array(stack)


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
                continue

            # Load the images
            img = self._load_stack(str(input_image_path)).astype(np.float32)
            gt_raw = self._load_stack(str(groundtruth_path)).astype(np.int64)
            gt = np.zeros_like(gt_raw)
            gt[gt_raw > 128] = 255

            # Transpose
            img = img.transpose(2, 0, 1)  # [x, y, z] -> [z, x, y]
            gt = gt.transpose(2, 0, 1)

            # Cut into 64x64x64 chunks
            z, x, y = img.shape
            for i in range(0, z, 64):
                for j in range(0, x, 64):
                    for k in range(0, y, 64):
                        img_chunk = img[i:i+64, j:j+64, k:k+64]
                        gt_chunk = gt[i:i+64, j:j+64, k:k+64]

                        # Ensure the chunk is 64x64x64
                        if img_chunk.shape == (64, 64, 64) and gt_chunk.shape == (64, 64, 64):
                            # Expand dimensions
                            img_chunk = torch.from_numpy(np.ascontiguousarray(img_chunk)).unsqueeze(0)
                            gt_chunk = torch.from_numpy(np.ascontiguousarray(gt_chunk)).unsqueeze(0)

                            # Normalize
                            img_chunk = img_chunk / 255.0
                            gt_chunk = gt_chunk // 255

                            images.append(img_chunk)
                            groundtruth.append(gt_chunk)

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
        # return self.images[idx], self.groundtruth[idx]
        return self.images[idx], self.groundtruth[idx]

