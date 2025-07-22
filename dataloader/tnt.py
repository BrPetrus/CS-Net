from __future__ import print_function, division
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance
from utils.misc import ReScaleSize
import random
import warnings
import numpy as np
import tifffile

warnings.filterwarnings('ignore')


def load_dataset(root_dir, train=True, inference=False):
    images = []
    groundtruth = []
    if train:
        sub_dir = 'training'
    elif inference:
        sub_dir = "inference"
    else:
        sub_dir = 'test'
    images_path = os.path.join(root_dir, sub_dir, 'images')
    groundtruth_path = os.path.join(root_dir, sub_dir, 'masks')

    for file in glob.glob(os.path.join(images_path, '*.tif')):
        image_name = os.path.basename(file)
        groundtruth_name = f'mask{image_name[1:]}'

        # Read the files and split
        img = tifffile.imread(os.path.join(images_path, image_name))
        if not inference:
            mask = tifffile.imread(os.path.join(groundtruth_path, groundtruth_name))
        split_imgs = []
        split_masks = []
        for z_idx in range(img.shape[0]):
            split_imgs.append(img[z_idx, ...])
            if not inference:
                split_masks.append(mask[z_idx, ...])
        

    print(f"[TNT DATALOADER] Found {len(split_imgs)} imgs")
    if inference:
        return split_imgs, []
    return split_imgs, split_masks

class TNTData(Dataset):
    def __init__(self,
                 root_dir,
                 train=True,
                 rotate=40,
                 flip=True,
                 random_crop=True,
                 scale1=688,
                 inference=False):

        self.inference = inference
        self.root_dir = root_dir
        self.train = train
        self.rotate = rotate
        self.flip = flip
        self.random_crop = random_crop
        self.transform = transforms.ToTensor()
        self.resize = scale1
        if self.inference and self.train:
            raise ValueError("Cannot set inference and training at the same time")
        self.images, self.groundtruth = load_dataset(self.root_dir, self.train, self.inference)

    def __len__(self):
        return len(self.images)

    def RandomCrop(self, image, label, crop_size):
        crop_width, crop_height = crop_size
        w, h = image.size
        left = random.randint(0, w - crop_width)
        top = random.randint(0, h - crop_height)
        right = left + crop_width
        bottom = top + crop_height
        new_image = image.crop((left, top, right, bottom))
        new_label = label.crop((left, top, right, bottom))
        return new_image, new_label

    def RandomEnhance(self, image):
        return image
        value = random.uniform(-2, 2)
        random_seed = random.randint(1, 4)
        if random_seed == 1:
            img_enhanceed = ImageEnhance.Brightness(image)
        elif random_seed == 2:
            img_enhanceed = ImageEnhance.Color(image)
        elif random_seed == 3:
            img_enhanceed = ImageEnhance.Contrast(image)
        else:
            img_enhanceed = ImageEnhance.Sharpness(image)
        image = img_enhanceed.enhance(value)
        return image

    def __getitem__(self, idx):
        # img_path = self.images[idx]
        # gt_path = self.groundtruth[idx]
        # image = Image.open(img_path)
        # label = Image.open(gt_path)
        # TODO: does not support full 16bit
        # TODO: WRONG!!!
        data = self.images[idx]
        data = (data - data.min()) / (data.max() - data.min())
        data *= 255
        image = Image.fromarray(np.repeat(data[..., np.newaxis], 3, axis=2).astype(np.uint8))
        if not self.inference:
            label = Image.fromarray((self.groundtruth[idx] > 0.0).astype(bool))
        image = ReScaleSize(image, self.resize)
        if not self.inference:
            label = ReScaleSize(label, self.resize)

        if self.train:
            # augumentation
            angel = random.randint(-self.rotate, self.rotate)
            image = image.rotate(angel)
            label = label.rotate(angel)

            if random.random() > 0.5:
                image = self.RandomEnhance(image)

            image, label = self.RandomCrop(image, label, crop_size=[self.resize, self.resize])

            # flip
            if self.flip and random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

        else:
            img_size = image.size
            if img_size[0] != self.resize:
                image = image.resize((self.resize, self.resize))
                if not self.inference:
                    label = label.resize((self.resize, self.resize))

        image = self.transform(image)
        if not self.inference:
            label = self.transform(label)
            return image, label
        return image  # Inference

        # import matplotlib.pyplot as plt
        # plt.imshow(np.transpose(image, (1, 2, 0)))
        # plt.figure()
        # plt.imshow(label[0])
        # plt.show()
        return image, label
