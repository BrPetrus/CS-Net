# -*- coding: utf-8 -*-
# @Author : Lei Mou
# @Modified by : Bruno Petrus 2024
# @File   : train3d.py

"""
Training script for CS-Net 3D
"""
# Standard Library
import os
import datetime
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# PyTorch
import torch
import torchsummary
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from model.csnet_3d import CSNet3D
from dataloader.VascuSynthLoader import Data

# Project files
from utils.train_metrics import metrics3d
from utils.losses import WeightedCrossEntropyLoss, DiceLoss
from utils.visualize import init_visdom_line, update_lines

from sklearn.model_selection import KFold

from typing import List

args = {
    'root'      : '/home/xpetrus/DP/CS-Net',
    'data_path' : '/home/xpetrus/DP/Datasets/External/VascuSynthMine02',
    # 'root'      : '/home/bruno/DP/CS-Net',
    # 'data_path' : '/home/bruno/DP/VascuSynth/dataset',
    'epochs'    : 100,
    'lr'        : 0.001,
    'snapshot'  : 100,
    'valid_step' : 5,
    'ckpt_path' : './checkpoint3D/',
    'batch_size': 4,
    'k_folds'   : 2,
    # 'weight_decay': 0.0005,
    'learning_rate_decay': 0.9,
}


def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    date = datetime.datetime.now().strftime("%Y-%m-%d-")
    torch.save(net, args['ckpt_path'] + 'CSNet3D_' + date + iter + '.pkl')
    print("{} Saved model to:{}".format("\u2714", args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr: float, iter: int, max_iter: int, power: float = 0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train_with_kfold() -> List[nn.Module]:
    """
    Train the CSNet3D model using K-Fold cross-validation.

    Returns:
        List[nn.Module]: List of trained models for each fold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k_folds = args['k_folds']
    kf = KFold(n_splits=k_folds, shuffle=True)

    nets = []

    # Load training data
    full_train_data = Data(args['data_path'], train=True)

    criterion = nn.CrossEntropyLoss().to(device)
    criterion2 = WeightedCrossEntropyLoss().to(device)
    criterion3 = DiceLoss(device=device).to(device)

    # Start training
    print("\033[1;30;44m {} Start training ... {}\033[0m".format("*" * 8, "*" * 8))

    # K Fold loop
    for fold, (train_index, validation_index) in enumerate(kf.split(full_train_data)):
        print(f"Fold: {fold}")
        net = CSNet3D(classes=2, channels=1).to(device)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net).to(device)
        else:
            net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['learning_rate_decay'])
        nets.append(net)

        # Create training and validation subsets for this fold
        train_subset = Subset(full_train_data, train_index)
        val_subset = Subset(full_train_data, validation_index)

        # DataLoaders for batching
        train_loader = DataLoader(train_subset, batch_size=args['batch_size'], num_workers=4, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args['batch_size'], num_workers=4, shuffle=False)

        iters = 1
        for epoch in range(args['epochs']):
            print(f"Epoch {epoch+1}/{args['epochs']} with learning rate {optimizer.param_groups[0]['lr']}")
            net.train()
            for idx, batch in enumerate(train_loader):
                # Get the data
                image = batch[0].to(device)
                label = batch[1].to(device)
                
                # Run the model
                optimizer.zero_grad()
                pred = net(image)

                # Calculate the loss
                loss_dice = criterion3(pred, label)
                label = label.squeeze(1)
                loss_ce = criterion(pred, label)
                loss_wce = criterion2(pred, label)
                loss = (loss_ce + 0.6 * loss_wce + 0.4 * loss_dice) / 3
                loss.backward()
                optimizer.step()

                # Print the metrics
                tp, fn, fp, iou = metrics3d(pred, label, pred.shape[0])
                print(
                    '\033[1;36m [{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tTP:{3:.4f}\tFN:{4:.4f}\tFP:{5:.4f}\tIoU:{6:.4f} '.format(
                        epoch + 1, iters, loss.item(), tp / pred.shape[0], fn / pred.shape[0], fp / pred.shape[0],
                        iou / pred.shape[0]))
                iters += 1

            # adjust_lr(optimizer, base_lr=args['lr'], iter=global_iters, max_iter=args['epochs'] * len(train_loader), power=args['learning_rate_decay'])

            if (epoch + 1) % args['snapshot'] == 0:
                save_ckpt(net, str(epoch + 1))


            # Run on validation set
            with torch.no_grad():
                net.eval()
                tp, fn, fp, iou, loss = 0, 0, 0, 0, 0
                for idx, batch in enumerate(val_loader):
                    image = batch[0].to(device)
                    label = batch[1].to(device)
                    pred = net(image)
                    loss_dice = criterion3(pred, label)
                    label = label.squeeze(1)
                    loss_ce = criterion(pred, label)
                    loss_wce = criterion2(pred, label)
                    loss += ((loss_ce + 0.6 * loss_wce + 0.4 * loss_dice) / 3).item()
                    # tp, fn, fp, iou = metrics3d(pred, label, pred.shape[0])
                    tp_batch, fn_batch, fp_batch, iou_batch = metrics3d(pred, label, pred.shape[0])
                    tp += tp_batch
                    fn += fn_batch
                    fp += fp_batch
                    iou += iou_batch
                tp /= len(val_loader)
                fn /= len(val_loader)
                fp /= len(val_loader)
                iou /= len(val_loader)
                loss /= len(val_loader)

                print(
                    f'Validation loss: {loss} TP: {tp} FN: {fn} FP: {fp} IoU: {iou}'
                )

        # Save the final model
        save_ckpt(net, f'final_fold-{fold}')
    
    return nets


def predict(net, device, output_dir):
    print("\033[1;30;43m {} Model evaluation ... {}\033[0m".format("*" * 8, "*" * 8))
    test_data = Data(args['data_path'], train=False)
    batchs_data = DataLoader(test_data, batch_size=1)
    # path = os.path.join(output_dir, 'results')
    # os.makedirs(path)

    TP, FN, FP, IoU = [], [], [], []
    file_num = 0
    net.eval()
    with torch.no_grad():
        for idx, batch in enumerate(batchs_data):
            path = os.path.join(output_dir, f'idx-{idx}')
            os.makedirs(path)

            image = batch[0].float().to(device)
            label = batch[1].to(device)
            pred_val = net(image)
            label = label.squeeze(1)

            fig, ax = plt.subplots(3, 1)
            ax[0].imshow(image[0, 0, :, :, 32].cpu().numpy())
            ax[1].imshow(pred_val[0, 1, :, :, 32].cpu().numpy())
            ax[2].imshow(label[0, :, :, 32].cpu().numpy())
            fig.savefig(os.path.join(path, 'result.png'))
            plt.show()

            # Now save the whole stacks
            image_stack = image[0, 0].cpu().numpy()
            pred_stack = pred_val[0, 1].cpu().numpy()
            label_stack = label[0].cpu().numpy() * 255

            tiff.imwrite(os.path.join(path, f'image_stack_{idx}.tiff'), image_stack)
            tiff.imwrite(os.path.join(path, f'pred_stack_{idx}.tiff'), pred_stack)
            tiff.imwrite(os.path.join(path, f'label_stack_{idx}.tiff'), label_stack)
            

            # loss = criterion(pred_val, label)
            tp, fn, fp, iou = metrics3d(pred_val, label, pred_val.shape[0])
            print(
                "--- test TP:{0:.4f}    test FN:{1:.4f}    test FP:{2:.4f}    test IoU:{3:.4f}".format(tp, fn, fp, iou))
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
            IoU.append(iou)
            file_num += 1
    return np.mean(TP), np.mean(FN), np.mean(FP), np.mean(IoU)




if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nets = train_with_kfold() 
    for k_fold_idx, net in enumerate(nets):
        output_dir = os.path.join('.', current_time, f'{k_fold_idx}-fold')
        predict(net, 'cuda', output_dir)
