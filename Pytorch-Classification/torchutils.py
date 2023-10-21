# 数据增强和测试指标的代码集中在这里

# 导入必备的包
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import math
# 网络模型构建需要的包
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_score
# Metric 测试准确率需要的包
from sklearn.metrics import f1_score, accuracy_score, recall_score
# Augmentation 数据增强要使用到的包
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import datasets, models, transforms

# 这个库主要用于定义如何进行数据增强。
# https://zhuanlan.zhihu.com/p/149649900?from_voters_page=true
def get_torch_transforms(img_size=224):
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation((-5, 5)),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

# 定义计算准确率的函数
def accuracy(output, target):
    # 使用softmax函数计算输出的概率分布，dim=1表示在每行上执行softmax
    y_pred = torch.softmax(output, dim=1)

    # 使用argmax函数找到每行中概率最高的类别索引，同时将结果移到CPU上
    y_pred = torch.argmax(y_pred, dim=1).cpu()

    # 将目标数据移到CPU上
    target = target.cpu()

    # 使用accuracy_score函数计算预测的准确率并返回
    return accuracy_score(target, y_pred)


# 计算F1分数（宏平均）
def calculate_f1_macro(output, target):
    # 使用softmax函数计算输出的概率分布，dim=1表示在每行上执行softmax
    y_pred = torch.softmax(output, dim=1)

    # 使用argmax函数找到每行中概率最高的类别索引，同时将结果移到CPU上
    y_pred = torch.argmax(y_pred, dim=1).cpu()

    # 将目标数据移到CPU上
    target = target.cpu()

    # 使用f1_score函数计算F1分数，采用宏平均（average='macro'）
    return f1_score(target, y_pred, average='macro')

# 计算召回率（宏平均）
def calculate_recall_macro(output, target):
    # 使用softmax函数计算输出的概率分布，dim=1表示在每行上执行softmax
    y_pred = torch.softmax(output, dim=1)

    # 使用argmax函数找到每行中概率最高的类别索引，同时将结果移到CPU上
    y_pred = torch.argmax(y_pred, dim=1).cpu()

    # 将目标数据移到CPU上
    target = target.cpu()

    # 使用recall_score函数计算召回率，采用宏平均（average="macro"）
    # zero_division=0 参数用于处理分母为零的情况
    return recall_score(target, y_pred, average="macro", zero_division=0)


# 训练的时候输出信息使用
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


# 调整学习率
def adjust_learning_rate(optimizer, epoch, params, batch=0, nBatch=None):
    """ adjust learning of a given optimizer and return the new learning rate """
    new_lr = calc_learning_rate(epoch, params['lr'], params['epochs'], batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


""" learning rate schedule """


# 计算学习率
def calc_learning_rate(epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type='cosine'):
    if lr_schedule_type == 'cosine':
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError('do not support: %s' % lr_schedule_type)
    return lr
