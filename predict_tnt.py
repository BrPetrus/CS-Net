import torch
from torchvision import transforms
from PIL import Image

import numpy as np
import os
from dataloader.tnt import load_dataset, TNTData
from torch.utils.data import DataLoader

DATABASE = '/home/xpetrus/DP/Datasets/TNT_data/tnt_dataset_csnet_2dsplit'
args = {
    'root': DATABASE,
    'test_path': DATABASE + 'test/',
    'pred_path': './predictions/',
    'img_size' : 512,
    'checkpoint_path': "./checkpoint/CS_Net_TNT_fold-1_100.pkl"
}

if not os.path.exists(args['pred_path']):
    os.makedirs(args['pred_path'])


def rescale(img):
    w, h = img.size
    min_len = min(w, h)
    new_w, new_h = min_len, min_len
    scale_w = (w - new_w) // 2
    scale_h = (h - new_h) // 2
    box = (scale_w, scale_h, scale_w + new_w, scale_h + new_h)
    img = img.crop(box)
    return img


def load_tnt():
    return TNTData(
        DATABASE,
        False,
    )
    


def load_net(checkpoint_path: str):
    torch.serialization.add_safe_globals('torch.nn.parallel.data_parallel.DataParallel')
    net = torch.load(checkpoint_path, weights_only=False)
    return net


def save_imgs(pred, prefix=''):
    save_path = args['pred_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if pred.ndim != 3:
        raise ValueError("Expected an array of 2D grayscale images")
    for z_idx in range(pred.shape[0]):
        img = Image.fromarray(pred[z_idx, ...])
        img.save(os.path.join(save_path, f"{prefix}_{z_idx}.tif"))

def predict(device="cpu"):
    net = load_net(args['checkpoint_path']).to(device)
    dataset = load_tnt()
    dataloader = DataLoader(dataset, 8, False)

    net.eval()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for batch_idx, (img, mask) in enumerate(dataloader):
            img_dev = img.to(device)
            output = torch.sigmoid(net(img_dev)).detach().cpu()
            thresholded = output > 0.5
            mask = mask.cpu()
            TP += ((thresholded == True) & (mask == True)).float().sum()
            TN += ((thresholded == False) & (mask == False)).float().sum()
            FP += ((thresholded == True) & (mask == False)).float().sum()
            FN += ((thresholded == False) & (mask == True)).float().sum()

            # Save the predictions
            print(f"output shape {output.shape}")
            print(f"mask shape {mask.shape}")
            print(f"max value in batch data {img.max()}")
            print(f" in label {mask.max()}")
            save_imgs(output.squeeze(1).numpy(), str(batch_idx))  # remove fake channel
            save_imgs(thresholded.squeeze(1).numpy(), f'{batch_idx}-thresh')

    
    # Evaluate
    total = TP + TN + FP + FN

    accuracy = (TP+TN) / total if total != 0 else 0.0
    precision = TP / (TP+FP) if TP+FP != 0 else 0.0
    recall = TP / (TP+FN) if TP+FN != 0 else 0.0

    print(f"Metrics:\n"
          f"Accuracy: {accuracy*100:.2f}%\n"
          f"Precision: {precision*100:.2f}%\n"
          f"Recall: {recall*100:.2f}%")
    print(f"Metrics:"
          f"Accuracy: {accuracy*100:.2f}%"
          f"Precision: {precision*100:.2f}%"
          f"Recall: {recall*100:.2f}%")
       



if __name__ == '__main__':
    predict()
