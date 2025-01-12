"""
训练器模块
"""
import os

import matplotlib.pyplot as plt
from core.Resunet import ResUnetPlusPlus
from model import ResUNet, UNet, SwinTransformerSys
import torch
from datasets import CVPPPDatasets
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import random
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

epoch = 500

def get_0_1_array(array, rate=0.2):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
    new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0 #将一部分换为0
    np.random.shuffle(new_array)#将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
    return re_array

def get_train_valid_sampler(trainset, valid=0.2):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def dice_coeff(pred, target):
    smooth = 1
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = 1.5 * (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth) * 100

class IOUMetric(object):
     """
     Class to calculate mean-iou using fast_hist method
     """
     def __init__(self, num_classes):
         self.num_classes = num_classes
         self.hist = np.zeros((num_classes, num_classes))

     def _fast_hist(self, label_pred, label_true):
         mask = (label_true >= 0) & (label_true < self.num_classes)
         hist = np.bincount(
             self.num_classes * label_true[mask].astype(int) +
             label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
         return hist

     def add_batch(self, predictions, gts):
         for lp, lt in zip(predictions, gts):
             self.hist += self._fast_hist(lp.flatten(), lt.flatten())

     def evaluate(self):
         acc = np.diag(self.hist).sum() / self.hist.sum()
         acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
         acc_cls = np.nanmean(acc_cls)
         iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
         mean_iu = np.nanmean(iu)
         freq = self.hist.sum(axis=1) / self.hist.sum()
         fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
         return acc, acc_cls, iu, mean_iu, fwavacc

def get_RGBD_loaders(batch_size=64, valid=0.2, num_workers=0, pin_memory=False):
    trainset = CVPPPDatasets()

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              drop_last=False)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              )

    return train_loader, valid_loader

def dice_loss(target, predictive, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = intersection / union
    return loss * 100


model = ResUnetPlusPlus(3)
# model = SwinTransformerSys(img_size=224, num_classes=3)
if torch.cuda.is_available():
    model.to('cuda')

loss_fn = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

train_loader, valid_loader = get_RGBD_loaders(batch_size=8, valid=0.1, num_workers=0, pin_memory=False)

def fit(epoch, model, train_loader, valid_loader):
    correct = 0
    total = 0
    running_loss = 0
    epoch_iou = []
    epoch_dice = []

    model.train()
    for x, y in train_loader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        print("y, y_pred", y_pred.size(), y.size())
        # mask = torch.tensor(get_0_1_array(np.array(x.cpu()))).to('cuda')
        # mask_img = x * mask
        # mask_re = model(mask_img.to(torch.float32))
        # 输入的图像，取第一张
        # image = x[0]
        # # 生成的图像，取第一张
        # x_ = y_pred[0]
        # # 标签的图像，取第一张
        # y_ = y[0]
        # # 三张图，从第0轴拼接起来，再保存
        # img = torch.stack([image, x_, y_], 0)
        # save_image(img.cpu(), os.path.join("./train_img", f"{epoch}.png"))
        loss = loss_fn(y_pred, y)
        # loss2 = loss_fn(mask_re, y)
        # loss = loss1 + 0.3 * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
            dice = dice_coeff(y, y_pred)
            intersection = torch.logical_and(y, y_pred)
            union = torch.logical_or(y, y_pred)
            batch_iou = (1 - torch.sum(intersection) / torch.sum(union)) * 100
            epoch_iou.append(batch_iou.item())  # 加了item
            epoch_dice.append(dice.item())

    # exp_lr_scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)

    test_total = 0
    test_running_loss = 0
    epoch_test_iou = []
    epoch_test_dice = []

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            # y_pred = torch.argmax(y_pred, dim=1)
            test_total += y.size(0)
            test_running_loss += loss.item()
            dice = dice_coeff(y, y_pred)
            epoch_test_dice.append(dice.item())
            intersection = torch.logical_and(y, y_pred)
            union = torch.logical_or(y, y_pred)
            batch_iou = (1 - torch.sum(intersection) / torch.sum(union)) * 100
            epoch_test_iou.append(batch_iou.item())  # 加了item

    epoch_test_loss = test_running_loss / len(valid_loader.dataset)

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'train_MIOU:', round(np.mean(epoch_iou), 3),
          'train_DICE', round(np.mean(epoch_dice), 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_MIOU:', round(np.mean(epoch_test_iou), 3),
          'test_DICE', round(np.mean(epoch_test_dice), 3)
          )

    return epoch_loss, epoch_test_loss

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epoch):
    epoch_loss, epoch_test_loss = fit(epoch,
                                                                 model,
                                                                 train_loader,
                                                                 valid_loader)
    train_loss.append(epoch_loss)
    test_loss.append(epoch_test_loss)