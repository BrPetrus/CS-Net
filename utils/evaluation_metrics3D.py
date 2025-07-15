#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                                             ║
# ║        __  __                                        ____                __                                 ║
# ║       /\ \/\ \                                      /\  _`\             /\ \  __                            ║
# ║       \ \ \_\ \     __     _____   _____   __  __   \ \ \/\_\    ___    \_\ \/\_\    ___      __            ║
# ║        \ \  _  \  /'__`\  /\ '__`\/\ '__`\/\ \/\ \   \ \ \/_/_  / __`\  /'_` \/\ \ /' _ `\  /'_ `\          ║
# ║         \ \ \ \ \/\ \L\.\_\ \ \L\ \ \ \L\ \ \ \_\ \   \ \ \L\ \/\ \L\ \/\ \L\ \ \ \/\ \/\ \/\ \L\ \         ║
# ║          \ \_\ \_\ \__/.\_\\ \ ,__/\ \ ,__/\/`____ \   \ \____/\ \____/\ \___,_\ \_\ \_\ \_\ \____ \        ║
# ║           \/_/\/_/\/__/\/_/ \ \ \/  \ \ \/  `/___/> \   \/___/  \/___/  \/__,_ /\/_/\/_/\/_/\/___L\ \       ║
# ║                              \ \_\   \ \_\     /\___/                                         /\____/       ║
# ║                               \/_/    \/_/     \/__/                                          \_/__/        ║
# ║                                                                                                             ║
# ║           49  4C 6F 76 65  59 6F 75 2C  42 75 74  59 6F 75  4B 6E 6F 77  4E 6F 74 68 69 6E 67 2E            ║
# ║                                                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# @Author : Lei Mou
# @File   : evaluation_metrics3D.py
import numpy as np
import SimpleITK as sitk
import glob
import os
import torch
from scipy.spatial import distance
from sklearn.metrics import f1_score
from dataclasses import dataclass


def numeric_score(pred, gt):
    FP = float(np.sum((pred == 255) & (gt == 0)))
    FN = float(np.sum((pred == 0) & (gt == 255)))
    TP = float(np.sum((pred == 255) & (gt == 255)))
    TN = float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def Dice(pred, gt):
    pred = np.int64(pred / 255)
    gt = np.int64(gt / 255)
    dice = np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))
    return dice


def IoU(pred, gt):
    pred = np.int64(pred / 255)
    gt = np.int64(gt / 255)
    m1 = np.sum(pred[gt == 1])
    m2 = np.sum(pred == 1) + np.sum(gt == 1) - m1
    iou = m1 / m2
    return iou


@dataclass(frozen=True)
class Metrics3D:
    """ Dataclass to store the metrics for 3D images """
    TP: float
    FN: float
    FP: float
    TN: float
    TPR: float
    FNR: float
    FPR: float
    IoU: float
    Acc: float
    Sen: float
    Spe: float
    Dice: float
    F1: float

def metrics_3d(pred, gt) -> Metrics3D:
    """ Calculates the metrics for 3D images """

    # pred = (pred.detach().cpu().numpy() > 0.5).astype(np.uint8)
    # pred = torch.argmax(pred, dim=1)
    # output = np.zeros_like(pred, dtype=np.int32)
    # gt = np.array(gt, dtype=np.int32)
    # output[pred > 0.5] = 1
    # outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    # labels = (gt.data.cpu().numpy() * 255).astype(np.uint8)

    # aux = (pred.detach().cpu().numpy() > 0.5)
    # outputs = np.array(aux*255, dtype=np.int32)
    # labels = np.array(gt.detach().cpu().numpy()*255, dtype=np.int32)
    outputs = (pred > 0.5).astype(np.int32) * 255
    labels = (gt * 255).astype(np.int32)

    FP, FN, TP, TN = numeric_score(outputs, labels)
    tpr = TP / (TP + FN + 1e-10)
    fnr = FN / (FN + TP + 1e-10)
    fpr = FN / (FP + TN + 1e-10)
    iou = TP / (TP + FN + FP + 1e-10)  # TODO: rename to Jaccard Index
    Acc = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    Sen = TP / (TP + FN + 1e-10)
    Spe = TN / (TN + FP + 1e-10)
    Dice = 2 * TP / (2 * TP + FP + FN + 1e-10)
    F1 = f1_score(labels.flatten(), outputs.flatten(), pos_label=255)
    return Metrics3D(TP, FN, FP, TN, tpr, fnr, fpr, iou, Acc, Sen, Spe, Dice, F1)


def over_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float(np.sum(gt == 255))
    Os = np.float(np.sum((pred == 255) & (gt == 0)))
    OR = Os / (Rs + Os)
    return OR


def under_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float(np.sum(gt == 255))
    Us = np.float(np.sum((pred == 0) & (gt == 255)))
    Os = np.float(np.sum((pred == 255) & (gt == 0)))
    UR = Us / (Rs + Os)
    return UR
