import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from collections import deque
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import h5py
import random
from data import imageDataset
from loss import SupConLoss
from model import SupConClipResNet
from util import  AverageMeter,save_model
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




path='/blue/npadillacoreano/yidaiyao/cross-subject/'
W=9
hdf5_file ='seq_mouse_30_window_9_516session.hdf5'#'seq_mouse_30_window_9.hdf5'
data = h5py.File(hdf5_file, 'r')
data.keys()
arg=5
alllabels=deque(data['image_v1'].keys())*arg
drop=3
newW=W-drop
batch_size=256
temperature=0.07
    
imagedataset=imageDataset(alllabels,data,W,newW)
loader = DataLoader(
    imagedataset,
    batch_size=batch_size
)
criterion = SupConLoss(device=device)

model=SupConClipResNet(in_channel=newW,name1='resnet18', name2='neural_resnet18',flag='image')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model= nn.DataParallel(model)
model.to(device)

def train(loader, model, criterion, optimizer, epoch,print_freq=50):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    end = time.time()
    for batch_idx, (image1,image2) in enumerate(loader):###same image with different argumentation 
        
        data_time.update(time.time() - end)
        
        images = torch.cat([image1, image2],dim=0)
        features = model(images.to(device).float())

        bs=image1.shape[0]
        f1, f2 = torch.split(features, [bs, bs], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features)
        
        losses.update(loss.item(), bs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        ###print
        if (batch_idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, batch_idx + 1, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


epochs=500
save_freq=50
save_folder='hpg_model/'

for epoch in range(1, epochs + 1):
    # adjust_learning_rate(opt, optimizer, epoch) ###later

    # train for one epoch
    time1 = time.time()
    loss = train(loader, model, criterion, optimizer, epoch)
    time2 = time.time()
    print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


    if epoch % save_freq == 0:
        save_file = os.path.join(
            save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        save_model(model, optimizer, epoch, save_file)

# save the last model
save_file = os.path.join(
    save_folder, 'last.pth')
save_model(model, optimizer, epochs, save_file)