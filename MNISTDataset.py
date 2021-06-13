#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import random

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images,  labels, transform):
        self.X = images
        self.y = labels
        self.transform = transform
        
    def __len__(self):
        return (len(self.X))

    def update(self, X_new, idx):
        self.X.iloc[idx, :] = X_new

    def get_sample(self, batch_size):

        n = self.__len__()
        idx = np.random.randint(1, n, batch_size)
      
        data = self.X.iloc[idx, :]
        data = torch.tensor(np.asarray(data).astype(np.uint8).reshape(batch_size, 1 , 28, 28))
        data = data.to(device).float ()

        target = self.y.iloc[idx]
        target = torch.tensor(np.asarray(target).astype(np.uint8))
        target = target.to(device)

        return data, target, idx

    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        data = np.asarray(data).astype(np.uint8).reshape(28, 28, 1)
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data    

