# -*- coding: utf-8 -*-

from __future__ import division
import os
import numpy as np
import pandas as pd
import cv2 
import matplotlib.pyplot as plt 
from glob import glob
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as ff
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torchvision import models
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import ssl
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.metrics import confusion_matrix
import numpy as np
import six
from sklearn.model_selection import train_test_split


# data_dir = "/content/kaggle_3m"   
data_dir = "/home/bg2502/HPML_Project/lgg-mri-segmentation/kaggle_3m"


class LabelProcessor:

    def __init__(self):
        self.colormap = self.read_color_map()
        self.cm2lbl = self.encode_label_pix(self.colormap)
    
    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')
    
    @staticmethod
    def read_color_map():  
        colormap = []
        colormap.append([0,0,0])
        colormap.append([255,255,255])
        return colormap
    
    @staticmethod
    def encode_label_pix(colormap):     
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl


class MRIDataset(Dataset):

    
    def __init__(self, img_path, label_path):
        
        if not isinstance(img_path, np.ndarray):
            self.img_path = np.array(img_path)
            self.label_path = np.array(label_path)
        self.labelProcessor = LabelProcessor()

    def __getitem__(self, index):
        img = self.img_path[index]
        label = self.label_path[index]
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        # transform
        img, label = self.img_transform(img, label)

        return {'img': img, 'label': label}

    def __len__(self):
        return len(self.img_path)

    def img_transform(self, img, label):
        # 对图片和标签做一些数值处理
        transform_img = transforms.Compose([transforms.ToTensor(),  # 转tensor
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        img = transform_img(img)
        label = self.labelProcessor.encode_label_img(label)
        label = torch.from_numpy(label)

        return img, label


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out



def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    pred_labels = pred_labels.flatten()
    gt_labels = gt_labels.flatten()
    confusion = confusion_matrix(gt_labels, pred_labels)
    if len(confusion)!= 2:
        confusion =  np.array([confusion[0][0],0,0,0]).reshape(2,2)
    return confusion


def calc_semantic_segmentation_iou(confusion):
    intersection = np.diag(confusion)
    union = np.sum(confusion, axis=1) + np.sum(confusion, axis=0) - np.diag(confusion)
    Ciou = (intersection / (np.maximum(1.0, union)+  1e-10) )
    mIoU = np.nanmean(Ciou)
    return mIoU

def calc_semantic_segmentation_dice(confusion):
    a, b = confusion
    tn, fp = a
    fn, tp = b
    return np.nanmean(2*tp/(2*tp + fn + fp+  1e-10))

def eval_semantic_segmentation(pred_labels, gt_labels):
    confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)
    mIoU = calc_semantic_segmentation_iou(confusion) 
    pixel_accuracy = np.nanmean(np.diag(confusion) / (confusion.sum(axis=1)+1e-10))
    class_accuracy = np.diag(confusion) / ( confusion.sum(axis=1) +  1e-10 )
    dice = calc_semantic_segmentation_dice(confusion)

    return {'miou': mIoU,
            'dice': dice}


"""# train"""

import time


def train(rank, world_size, net, batch_size, epochs, Load_train):
    torch.manual_seed(0)
    device = torch.device(f'cuda:{rank}')
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    net = net.to(device)

    net = net.train()
    net.to(device)
    net = DDP(net, device_ids=[rank])

    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
    Load_train,
    num_replicas=world_size,
    rank=rank)
    
    train_data = torch.utils.data.DataLoader(
        Load_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2
    )

    Epoch = 2

    train_miou_epoch = []
    train_dice_epoch = []


    test_miou_epoch = []
    test_dice_epoch = []

    # 训练轮次
    for epoch in range(Epoch):
        # xxx time
        
        comm_time = 0

        train_loss = 0
        train_miou = 0
        train_dice = 0
        error = 0
        print('Epoch is [{}/{}], batch size {}'.format(epoch + 1, Epoch, batch_size))
        epoch_start_time = time.time()
        trainloader_iter = iter(train_data)
        # 训练批次
        for i in range(len(train_data)):
            start_comm = time.time()
            sample = next(trainloader_iter)
            img_data = sample['img'].to(device)
            img_label = sample['label'].to(device)
            # xxx time
            out = net(img_data)
            comm_time += time.time() - start_comm

            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            true_label = img_label.data.cpu().numpy()
            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_miou += eval_metrix['miou']
            train_dice += eval_metrix['dice']


        epoch_time = time.time() - epoch_start_time
        print(f"Training time: {epoch_time:.2f} seconds")
        print(f"Communication time: {comm_time:.4f} seconds")
        print("-----------------")

    

        train_miou_epoch.append(train_miou / len(train_data))
        train_dice_epoch.append(train_dice / len(train_data))




if __name__ == "__main__":
    num_class = 2
    images_dir = []
    masks_dir = []
    masks_dir = glob(data_dir + '/*/*_mask*')

    for i in masks_dir:
        images_dir.append(i.replace('_mask',''))

    Load_train = MRIDataset(images_dir, masks_dir)

    gpu_count = 1
    epochs = 2
    batch_sizes = [16, 32, 64, 128, 256]
    gpu_counts = [1,2,3,4]
    for batch_size in batch_sizes:
        try:
            net =  U_Net(3,2)
            print("U_Net")
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            for gpu_count in gpu_counts:        
                print(f"\nRunning with {gpu_count} GPUs")
                world_size = gpu_count
                mp.spawn(train, args=(world_size, net, batch_size, epochs, Load_train), nprocs=world_size, join=True)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size} is too large for the available GPU memory.")
                break
            else:
                raise e 



