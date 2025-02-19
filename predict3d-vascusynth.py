# Standard Library
import os
import datetime
import numpy as np
from numpy.typing import NDArray
import tifffile as tiff
import matplotlib.pyplot as plt
from typing import List
import glob

# PyTorch
import torch
import torchsummary
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from model.csnet_3d import CSNet3D
from dataloader.VascuSynthLoader import Data

from PIL import Image

# Project files
from utils.train_metrics import metrics3d
from utils.losses import WeightedCrossEntropyLoss, DiceLoss
from utils.visualize import init_visdom_line, update_lines
from sklearn.model_selection import KFold
import re


def predict_raw(net, device, data, output_dir) -> None:
    """ Just raw predictions without any GT or metrics. """
    with torch.no_grad():
        net.eval()

        # Prepare
        data = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
        data /= data.max()
        data = data.to(device)

        output = net(data)
        # Save
        img = Image.fromarray(output)
        img.save('prediction.tif')


def load_180322(path: str) -> NDArray[np.float64]:
    # Find all files matching the pattern
    file_pattern = os.path.join(path, 's_C001Z*T*.tif')
    files = sorted(glob.glob(file_pattern))
    
    # # Split into T, Z categories
    # data_meta = []
    # pattern = r'c_C(\d+)Z(\d+)T(\d+)\.tif'
    # regex = re.compile(pattern)
    # for f in files:
    #     # match = re.search(pattern, os.path.basename(f))
    #     match = regex.search(os.path.basename(f))
    #     if match:
    #         C, Z, T = map(int, match.groups())
    #         data_meta.append((C, Z, T))

    # Load each file into a list of arrays
    arrays = [np.array(Image.open(f)) for f in files]
    
    # Stack arrays along a new axis (e.g., the first axis)
    data = np.stack(arrays, axis=0)
    
    data = data.reshape((1, 7, 20, 512, 512)) # C,Z,T,X,Y
    return data

def load_data(path: str) -> NDArray:
    pass


if __name__ == "__main__":
    # Load net
    net = torch.load('/home/xpetrus/DP/CS-Net/checkpoint3D/CSNet3D_2025-02-16-final_fold-1.pkl')

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = load_180322('/home/xpetrus/DP/Datasets/TNT_data/180322_Sqh-mCh Tub-GFP 16h_110.tif.files/')

    # Illustrate few images
    fig, ax = plt.subplots()
    t, z, c = 6-1, 5-1, 1-1
    img = data[c, z, t]  # Singl image
    ax.imshow(img)
    fig.savefig('t6z5.png', dpi=600)

    # Predict
    predict_raw(net, device, img, '.')