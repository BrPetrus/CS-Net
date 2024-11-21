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
from dataloader.stare import Data
from utils.train_metrics import metrics
from utils.visualize import init_visdom_line, update_lines
from utils.dice_loss_single_class import dice_coeff_loss

from sklearn.model_selection import KFold

args = {
    'root'      : '/home/xpetrus/DP/CS-Net',
    'data_path' : '/home/xpetrus/DP/Datasets/External/STARE',
    'epochs'    : 2000,
    'lr'        : 0.0001,
    'snapshot'  : 100,
    'test_step' : 1,
    'ckpt_path' : 'checkpoint/',
    'batch_size': 2,
    'kfold'     : 5,
}

# # # Visdom---------------------------------------------------------
# X, Y = 0, 0.5  # for visdom
# x_acc, y_acc = 0, 0
# x_sen, y_sen = 0, 0
# env, panel = init_visdom_line(X, Y, title='Train Loss', xlabel="iters", ylabel="loss")
# env1, panel1 = init_visdom_line(x_acc, y_acc, title="Accuracy", xlabel="iters", ylabel="accuracy")
# env2, panel2 = init_visdom_line(x_sen, y_sen, title="Sensitivity", xlabel="iters", ylabel="sensitivity")
# # # ---------------------------------------------------------------

def save_ckpt(net, iter, kfold=None):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    path = args['ckpt_path'] + 'CS_Net_DRIVE_' 
    if kfold is not None:
        path += kfold
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
    k_folds = 4
    kf = KFold(n_splits=k_folds, shuffle=True)
    
    # set the channels to 3 when the format is RGB, otherwise 1.
    #net = CSNet(classes=1, channels=3).cuda()
    #net = nn.DataParallel(net).cuda()
    
    nets = []


    criterion = nn.MSELoss().cuda()


    print("---------------start training- with K Fold-----------------")
    # load train datasetfor fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    full_dataset = Data(args['data_path'], train=True)
    
    
    # K-Fold loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"fold {fold+1}/{k_folds}")
        
        init_fold_visualization(fold+1)

        # Create the nets
        # TODO: reuse the same memory
        net = CSNet(classes=1, channels=3).cuda()
        optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=0.0005)
        #net = nn.DataParallel(net).cuda()
        
        # Create training and validation subsets for this fold
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        # DataLoaders for batching
        train_loader = DataLoader(train_subset, batch_size=args['batch_size'], num_workers=2, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args['batch_size'], num_workers=2, shuffle=False)  # TODO: not used now
        
        # Reset model weights for this fold
        #net.apply(init_weights)  # Optional: Reset weights to avoid contamination
        
        # Training loop for this fold
        net.train()
        t=0
        for epoch in range(args['epochs']):
            print(f"Epoch {epoch + 1}/{args['epochs']}")
            for idx, batch in enumerate(train_loader):
                image = batch[0].cuda()
                label = batch[1].cuda()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                pred = net(image)
                
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
        
        # Validation loop for this fold
        #validate(net, val_loader, criterion, fold)
        fold_acc, fold_sens = model_eval(net)
        print(f"Fold {fold+1}/{k_folds}: acc={fold_acc}, sens={fold_sens}")
        nets.append((net, fold_acc, fold_sens))
    


def train():

    # set the channels to 3 when the format is RGB, otherwise 1.
    net = CSNet(classes=1, channels=3).cuda()
    net = nn.DataParallel(net).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=0.0005)
    critrion = nn.MSELoss().cuda()

    # Print the architecture
    summary(net)

    # critrion = nn.CrossEntropyLoss().cuda()
    print("---------------start training------------------")
    # load train dataset
    train_data = Data(args['data_path'], train=True)
    batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=2, shuffle=True)

    iters = 1
    accuracy = 0.
    sensitivty = 0.
    net.train()
    for epoch in range(args['epochs']):
        print(f"Epoch {epoch}/{len(range(args['epochs']))}")
        for idx, batch in enumerate(batchs_data):
            image = batch[0].cuda()
            label = batch[1].cuda()
            optimizer.zero_grad()
            pred = net(image)
            #pred = pred.squeeze_(1)
            #loss = (pred - label).mean()  # TODO: stupid debg prupose
            #loss.backward()
            #print(loss)
            #print(type(loss))
            #loss.backward()
            loss1 = critrion(pred, label)
            #loss1.backward()
            loss2 = dice_coeff_loss(pred, label)
            loss = loss1 + loss2
            #loss = loss1
            loss.backward()
            #loss = 
            
            optimizer.step()
            acc, sen = metrics(pred, label, pred.shape[0])
            #acc, sen = 1, 1  # TODO: DBG Stupid
            print('[{0:d}:{1:d}] --- loss:{2:.10f}\tacc:{3:.4f}\tsen:{4:.4f}'.format(epoch + 1,
                                                                                     iters, loss.item(),
                                                                                     acc / pred.shape[0],
                                                                                     sen / pred.shape[0]))
            iters += 1
            # # ---------------------------------- visdom --------------------------------------------------
            X, x_acc, x_sen = iters, iters, iters
            Y, y_acc, y_sen = loss.item(), acc / pred.shape[0], sen / pred.shape[0]
            update_lines(env, panel, X, Y)
            update_lines(env1, panel1, x_acc, y_acc)
            update_lines(env2, panel2, x_sen, y_sen)
            # # --------------------------------------------------------------------------------------------

        adjust_lr(optimizer, base_lr=args['lr'], iter=epoch, max_iter=args['epochs'], power=0.9)
        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, epoch + 1)

# TODO: WHAT?
        # model eval
#        if (epoch + 1) % args['test_step'] == 0:
#            test_acc, test_sen = model_eval(net)
#            print("Average acc:{0:.4f}, average sen:{1:.4f}".format(test_acc, test_sen))

#           if (accuracy > test_acc) & (sensitivty > test_sen):
#                save_ckpt(net, epoch + 1 + 8888888)
#                accuracy = test_acc
#                sensitivty = test_sen


def model_eval(net):
    print("Start testing model...")
    test_data = Data(args['data_path'], train=False)
    batchs_data = DataLoader(test_data, batch_size=1)

    net.eval()
    Acc, Sen = [], []
    file_num = 0
    for idx, batch in enumerate(batchs_data):
        image = batch[0].float().cuda()
        label = batch[1].float().cuda()
        pred_val = net(image)
        acc, sen = metrics(pred_val, label, pred_val.shape[0])
        print("\t---\t test acc:{0:.4f}    test sen:{1:.4f}".format(acc, sen))
        Acc.append(acc)
        Sen.append(sen)
        file_num += 1
        # for better view, add testing visdom here.
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
