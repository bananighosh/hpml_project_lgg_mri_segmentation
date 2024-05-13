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
data_dir = "/home/bg2502/project/lgg-mri-segmentation/kaggle_3m"


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
        transform_img = transforms.Compose([transforms.ToTensor(),  # 转tensor
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        img = transform_img(img)
        label = self.labelProcessor.encode_label_img(label)
        label = torch.from_numpy(label)

        return img, label


class SegNet(nn.Module):
    def __init__(self, classes=19):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1_size = x12.size()
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2_size = x22.size()
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3_size = x33.size()
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4_size = x43.size()
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5_size = x53.size()
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=x5_size)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=x4_size)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=x3_size)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=x2_size)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2, output_size=x1_size)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d



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
        num_workers=4
    )

    Epoch = 2

    train_miou_epoch = []
    train_dice_epoch = []


    test_miou_epoch = []
    test_dice_epoch = []

    for epoch in range(Epoch):
        # xxx time
        epoch_start_time = time.time()
        comm_time = 0

        train_loss = 0
        train_miou = 0
        train_dice = 0
        error = 0
        print('Epoch is [{}/{}], batch size {}'.format(epoch + 1, Epoch, batch_size))

        for i, sample in enumerate(train_data):
           
            img_data = sample['img'].to(device)
            img_label = sample['label'].to(device)
            # xxx time
            
            start_comm = time.time()
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

            if i%100 ==0:
                print('|batch[{}/{}]|batch_loss:{:.9f}|'.format(i + 1, len(train_data), loss.item()))

        metric_description = '|Train dice|: {:.5f}\n|Train Mean IoU|: {:.5f}'.format(
            train_dice / len(train_data),
            train_miou / len(train_data))

        epoch_time = time.time() - epoch_start_time
        print(metric_description)
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
    batch_sizes = [16]
    gpu_counts = [1,2,3,4]
    for batch_size in batch_sizes:
        try:
            net = SegNet()
            print("SegNet")
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
