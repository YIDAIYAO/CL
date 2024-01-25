import torch
import random 
import numpy as np
from collections import deque

from torch.utils.data import Dataset, DataLoader


class imageDataset(torch.utils.data.Dataset):
    def __init__(self, allindex,data,W,newW):
        'Initialization'
        self.data = data
        self.allindex = allindex
        self.W=W
        self.newW=newW
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.allindex)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.allindex[index]

        # Load data and get label
        image1 = np.array(self.data['image_v1'][ID])
        image2 = np.array(self.data['image_v2'][ID])
        image = np.array(np.concatenate((image1,image2),axis=-1)/255)
        keep1=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        keep2=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        
        return image[keep1],image[keep2]
    
class poseDataset(torch.utils.data.Dataset):
    def __init__(self, allindex,data,W,newW):
        'Initialization'
        self.data = data
        self.allindex = allindex
        self.W=W
        self.newW=newW
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.allindex)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.allindex[index]

        # Load data and get label
        pose = np.array(self.data['pose'][ID])/256 ##image size:(256,128)
        pose = pose.reshape(self.W,-1)
        pose=np.expand_dims(pose,axis=-1)
        
        # image2 = np.array(self.data['image_v2'][ID])
        # image = np.array(np.concatenate((image1,image2),axis=-1)/255)
        keep1=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        keep2=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        
        return pose[keep1],pose[keep2]
    
class neuralDataset(torch.utils.data.Dataset):
    def __init__(self, allindex,data,W,newW):
        'Initialization'
        self.data = data
        self.allindex = allindex
        self.W=W
        self.newW=newW
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.allindex)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.allindex[index]

        # Load data and get label
        neural = np.array(self.data['neural'][ID]) ##image size:(256,128)
        neural=np.expand_dims(neural,axis=-1)
        # image2 = np.array(self.data['image_v2'][ID])
        # image = np.array(np.concatenate((image1,image2),axis=-1)/255)
        keep1=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        keep2=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        
        return neural[keep1],neural[keep2]