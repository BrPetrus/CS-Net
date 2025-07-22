"""
Training script for CS-Net
"""
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
import visdom
import numpy as np
from model.csnet import CSNet
#from dataloader.stare import Data
from dataloader.tnt import TNTData
from utils.train_metrics import metrics
from utils.visualize import init_visdom_line, update_lines
from utils.dice_loss_single_class import dice_coeff_loss
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

args = {
    'root'      : '/home/xpetrus/DP/CS-Net',
    # 'data_path' : '/home/xpetrus/DP/Datasets/External/STARE',
    'data_path' : '/home/xpetrus/DP/Datasets/TNT_data/tnt_dataset_csnet_2dsplit',
    'epochs'    : 200,
    'lr'        : 0.0001,
    'snapshot'  : 100,
    'test_step' : 1,
    'ckpt_path' : 'checkpoint/',
    'batch_size': 4,
    'kfold'     : 2,
}

# # # Visdom---------------------------------------------------------
X, Y = 0, 0.5  # for visdom
x_acc, y_acc = 0, 0
x_sen, y_sen = 0, 0
env, panel = init_visdom_line(X, Y, title='Train Loss', xlabel="iters", ylabel="loss")
env1, panel1 = init_visdom_line(x_acc, y_acc, title="Accuracy", xlabel="iters", ylabel="accuracy")
env2, panel2 = init_visdom_line(x_sen, y_sen, title="Sensitivity", xlabel="iters", ylabel="sensitivity")
# # # ---------------------------------------------------------------

def save_ckpt(net, iter, kfold=None):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    path = args['ckpt_path'] + 'CS_Net_TNT_'
    if kfold is not None:
        path += str(kfold)
        path += "_"
    path += str(iter) + '.pkl'
    torch.save(net, path)
    print('--->saved model:{}<--- '.format(args['root'] + args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_with_kfodl():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    kf = KFold(n_splits=args['kfold'], shuffle=True)
    
    # set the channels to 3 when the format is RGB, otherwise 1.
    #net = CSNet(classes=1, channels=3).cuda()
    #net = nn.DataParallel(net).cuda()
    
    nets = []
    criterion = nn.MSELoss().to(device)


    print("---------------start training- with K Fold-----------------")
    # load train datasetfor fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    full_dataset = TNTData(args['data_path'], train=True)

    # Create the nets
    # TODO: reuse the same memory
    net = CSNet(classes=1, channels=3).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=0.0005)
    dataloader = DataLoader(full_dataset, args['batch_size'], shuffle=True)

    # Training loop for this fold
    init_fold_visualization(1)
    net.train()
    t=0
    fold=0
    import matplotlib.pyplot as plt
    for epoch in range(args['epochs']):
        print(f"Epoch {epoch + 1}/{args['epochs']}")
        for idx, batch in enumerate(dataloader):
            image = batch[0].to(device)
            label = batch[1].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred = net(image)
            
            # if epoch == 50 or epoch == 0:
            #     fig, ax = plt.subplots(ncols=2, nrows=2)
            #     ax = ax.flatten()
            #     print(image[0].shape)
            #     ax[0].imshow(torch.permute(image[0].detach().cpu(), (1, 2, 0)))
            #     ax[1].imshow(label[0, 0, ...].detach().cpu())
            #     ax[2].imshow(pred[0, 0, ...].detach().cpu())
            #     fig.tight_layout()
            #     fig.suptitle("0th index from batch")
            #     plt.savefig(f"epoch{epoch}.jpg")

            # Compute losses
            loss1 = criterion(pred, label)
            loss2 = dice_coeff_loss(pred, label)
            loss = loss1 + loss2
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Metrics
            acc, sen = metrics(pred, label, pred.shape[0])
            print('[Fold {0} Epoch {1} Batch {2}] --- Loss: {3:.10f}\tAcc: {4:.4f}\tSen: {5:.4f}'.format(
                fold + 1, epoch + 1, idx + 1, loss.item(), acc / pred.shape[0], sen / pred.shape[0]))


            # Update Visdom plots
            update_visdom_line(fold + 1, "loss", t, loss.item())
            update_visdom_line(fold + 1, "accuracy", t, acc/pred.shape[0])
            update_visdom_line(fold + 1, "sensitivity", t, sen/pred.shape[0])

            t += 1
        # Adjust learning rate
        adjust_lr(optimizer, base_lr=args['lr'], iter=epoch, max_iter=args['epochs'], power=0.9)

        # Save checkpoint at specified intervals
        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, epoch + 1, f"fold-{fold+1}")
    

    fold_acc, fold_sens = model_eval(net, device)
    print(f"Fold {fold+1}/{args['kfold']}: acc={fold_acc}, sens={fold_sens}")
    nets.append((net, fold_acc, fold_sens))
    save_ckpt(net, epoch, fold)


def model_eval(net, device):
    # raise RuntimeError("model_eval should not be called")
    print("Start testing model...")
    # return 0.0, 0.0
    test_data = TNTData(args['data_path'], train=False)
    batchs_data = DataLoader(test_data, batch_size=1)

    with torch.no_grad():
        net.eval()
        Acc, Sen = [], []
        file_num = 0
        for idx, batch in enumerate(batchs_data):
            image = batch[0].float().to(device)
            label = batch[1].float().to(device)
            pred_val = net(image)
            acc, sen = metrics(pred_val, label, pred_val.shape[0])
            print("\t---\t test acc:{0:.4f}    test sen:{1:.4f}".format(acc, sen))
            Acc.append(acc)
            Sen.append(sen)
            file_num += 1
            # for better view, add testing visdom here.
            fig, ax = plt.subplots(nrows=2, ncols=2)
            ax = ax.flatten()
            ax[0].imshow(torch.permute(image[0], (1, 2, 0)).cpu().detach())
            ax[1].imshow(label[0, 0].cpu().detach())
            ax[2].imshow(pred_val[0, 0].cpu().detach())
            ax[3].imshow(pred_val[0, 0].cpu().detach()*255 >= 100)
            fig.tight_layout()
            plt.savefig(f'{idx}.jpg')
    return np.mean(Acc), np.mean(Sen)

def init_visdom_line(X, Y, title, xlabel, ylabel, env_name):
    env = viz
    panel = env.line(
        X=[X], Y=[Y],
        opts=dict(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            showlegend=True
        ),
        env=env_name
    )
    return env, panel

# Track plots for each fold
fold_plots = {}

# Initialize Visdom lines for each fold
def init_fold_visualization(fold_id):
    fold_name = f"Fold_{fold_id}"  # Unique name for each fold
    fold_plots[fold_id] = {
        "loss": init_visdom_line(0, 0.5, title=f"Train Loss (Fold {fold_id})", xlabel="iters", ylabel="Loss", env_name=fold_name),
        "accuracy": init_visdom_line(0, 0, title=f"Accuracy (Fold {fold_id})", xlabel="iters", ylabel="Accuracy", env_name=fold_name),
        "sensitivity": init_visdom_line(0, 0, title=f"Sensitivity (Fold {fold_id})", xlabel="iters", ylabel="Sensitivity", env_name=fold_name),
    }

# Update Visdom plots dynamically
def update_visdom_line(fold_id, metric, x, y):
    env, panel = fold_plots[fold_id][metric]
    env_name=f"Fold_{fold_id}"
    env.line(
        X=[x], Y=[y],
        win=panel,
        update='append',
        env=env_name,
    )


if __name__ == '__main__':
    viz = visdom.Visdom()

    # Initialize a simple line
    env_name = "Test_Env"
    panel = viz.line(
        X=[0],
        Y=[0.5],
        opts=dict(
            title="Test Plot",
            xlabel="X-axis",
            ylabel="Y-axis",
            showlegend=True,
        ),
        env=env_name,
    )

    # Append new points
    for i in range(10):
        viz.line(
            X=[i],
            Y=[0.5 + i * 0.1],
            win=panel,
            update='append',
            env=env_name,
        )

    torch.autograd.set_detect_anomaly(True)
    train_with_kfodl()
