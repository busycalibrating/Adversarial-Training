#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import argparse
import os
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
import matplotlib.animation as animation

from IPython.display import HTML

from MNISTDataset import *
from mnist_net import *

epochs = 1
epsilon = 0.1
n_lan = 100
n_iter = 200
display = False

#mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
#mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
#train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
#test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)


path = "/content/sample_data/mnist_train_small.csv"
mnist_train_csv = pd.read_csv(path)

path = "/content/sample_data/mnist_test.csv"
mnist_test_csv = pd.read_csv(path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_labels = mnist_train_csv.iloc[:, 0]
train_images = mnist_train_csv.iloc[:, 1:]

train_labels_adv = mnist_train_csv.iloc[:, 0]
train_images_adv = mnist_train_csv.iloc[:, 1:]

test_labels = mnist_test_csv.iloc[:, 0]
test_images = mnist_test_csv.iloc[:, 1:]

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))
])

#Datasets

train_data = MNISTDataset(train_images,train_images_adv, train_labels, transform)
test_data = MNISTDataset(test_images,test_images, test_labels, transform)
# dataloaders
trainloader = DataLoader(train_data, batch_size=100, shuffle=True)
testloader = DataLoader(test_data, batch_size=100, shuffle=True)

#train_data_adv = MNISTDataset(train_images_adv, train_labels_adv, transform)
#trainloader_adv = DataLoader(train_data_adv, batch_size=100, shuffle=True)


torch.manual_seed(0)   

model_cnn = model_mnist()

if torch.cuda.is_available():
    model_cnn.cuda()
    
def Langevin(model, Z_adv, Z , y_i, n_lan,epsilon, step=0.1):
      samples = torch.tensor([]).to(device)
      img_list = []
      shape = Z_adv.shape
      Z_adv = Z_adv.to(device, dtype=torch.float)
      Z_adv.requires_grad_()

      Zj = Z_adv
      y = torch.tensor([]).to(device)
      y_i = y_i.to(device, dtype=torch.long)
      for i in range(n_lan):
        y = torch.cat((y,y_i),0)
        Zj = Zj.to(device, dtype=torch.float)
        Zj.requires_grad = True
        u = nn.CrossEntropyLoss()(model(Zj),y_i)
        grad = torch.autograd.grad(u, Zj)[0]
        tensor_step = torch.tensor(np.array([2 * step])).to(device)
        noise = torch.randn(shape).to(device)
        Zj = Zj.detach() - step * grad + torch.sqrt(tensor_step) * noise 
        Zj.data = torch.max(torch.min(Zj, Z+epsilon), Z-epsilon)
        Zj.data = Zj.clamp(0,1)
        #if i%20 == 0 : 
          #img_list.append(vutils.make_grid(Zj.detach().cpu(), padding=2, normalize=True))
        samples = torch.cat((samples,Zj.detach()),0)

        #samples.append(Zj.detach())

      return Zj.cpu().data.numpy(), samples, y

def epoch_adversarial_lan(train_data, model, n_lan, epsilon, n_iter, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for  i in range(n_iter):
        X, X_adv_prev , y, idx = train_data.get_sample(100)
        X_new,samples_lan,y_lan = Langevin(model,X_adv_prev,X, y,n_lan, epsilon, step=0.1)
        samples_lan  = samples_lan.to(device, dtype=torch.float)
        y_lan = y_lan.long()
        yp = model(samples_lan)
        loss = nn.CrossEntropyLoss()(yp,y_lan)

        total_err += (yp.max(dim=1)[1] != y_lan).sum().item()
        total_loss += loss.item() / (X.shape[0] * n_lan)
        print(total_err,total_loss)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        X_new = X_new.reshape(100,784)
        train_data.update_adv(pd.DataFrame(X_new).values, idx)
    return total_err / (train_data.__len__()*n_lan ), total_loss / train_data.__len__(), train_data

def train(epoch):
    opt = optim.SGD(model_cnn.parameters(), lr=0.01)
    model_cnn.cuda()
    train_errors = []
    train_losses = []
    for t in range(epoch):
        train_err, train_loss, data_adv = epoch_adversarial_lan(train_data, model_cnn, n_lan ,epsilon, n_iter, opt)
        train_errors.append(train_err)
        train_losses.append(train_loss)
        plot_adv(data_adv.X_adv.values,data_adv.__len__())
    if t == 4:
        for param_group in opt.param_groups:
               param_group["lr"] = 1e-2
    print(*("{:.6f}".format(i) for i in (train_errors)), sep="\t")
    print(*("{:.6f}".format(i) for i in (train_losses)), sep="\t")
    
    torch.save(model_cnn.state_dict(), "model_cnn.pt")
    
def plot_adv(images_values, size):
    images = np.reshape(images_values,(size,28,28))
    plt.figure(figsize=(10,10))
    for i in range (25):
        plt.subplot(5,5,i+1)
        plt.imshow(images[i,:,:])
        plt.axis('off')
    plt.subplots_adjust(wspace=0.0, hspace=0.1) 
    

train(epochs)

if display:
    plot_adv(data_adv.X_adv.values)

