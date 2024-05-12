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

"""# Load Dataset
"""

# data_dir = "/content/kaggle_3m"   
data_dir = "/home/sz3714/HPML_Project/lgg-mri-segmentation/kaggle_3m"


class LabelProcessor:

    def __init__(self):
        self.colormap = self.read_color_map()
        self.cm2lbl = self.encode_label_pix(self.colormap)
    
    # Label encoding, return the encoded label of 1 channel eg: [0000000][0011000][0000000]
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


class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        pretrained_net = models.vgg16_bn(weights="VGG16_BN_Weights.IMAGENET1K_V1")
        self.stage1 = pretrained_net.features[:7]
        self.stage2 = pretrained_net.features[7:14]
        self.stage3 = pretrained_net.features[14:24]
        self.stage4 = pretrained_net.features[24:34]
        self.stage5 = pretrained_net.features[34:]

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(512, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.conv_trans1 = nn.Conv2d(512, 256, 1)
        self.conv_trans2 = nn.Conv2d(256, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        # self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)

        self.upsample_2x_1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        # self.upsample_2x_1.weight.data = bilinear_kernel(512, 512, 4)

        self.upsample_2x_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        # self.upsample_2x_2.weight.data = bilinear_kernel(256, 256, 4)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)

        scores1 = self.scores1(s5)
        s5 = self.upsample_2x_1(s5)
        add1 = s5 + s4

        scores2 = self.scores2(add1)

        add1 = self.conv_trans1(add1)
        add1 = self.upsample_2x_2(add1)
        add2 = add1 + s3

        output = self.conv_trans2(add2)
        output = self.upsample_8x(output)
        return output



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
    net = DDP(net, device_ids=[rank],find_unused_parameters=True)

    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
    Load_train,
    num_replicas=world_size,
    rank=rank)
    
    train_data = torch.utils.data.DataLoader(
        Load_train,
        batch_size=batch_size,
        sampler=train_sampler
    )

    Epoch = 2

    train_miou_epoch = []
    train_dice_epoch = []


    test_miou_epoch = []
    test_dice_epoch = []

    # 训练轮次
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
            # 载入数据
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
            # 评估
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

    print("FCN")



    Load_train = MRIDataset(images_dir, masks_dir)

    gpu_count = 1
    epochs = 2
    batch_sizes = [16, 32, 64, 128, 256]
    gpu_counts = [1,2,3,4]
    for batch_size in batch_sizes:
        try:
            fcn = FCN(2)
            print("FCN")
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            for gpu_count in gpu_counts:        
                print(f"\nRunning with {gpu_count} GPUs")
                world_size = gpu_count
                mp.spawn(train, args=(world_size, fcn, batch_size, epochs, Load_train), nprocs=world_size, join=True)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size} is too large for the available GPU memory.")
                break
            else:
                raise e 
