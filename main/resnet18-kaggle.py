# -*- coding: utf-8 -*-

import os
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim 
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.models import resnet18, ResNet18_Weights

class LazyLoadDataSet(Dataset):
    ''' Lazily load image files and transform '''

    def __init__(self, path, train= True, transform = None):
        self.transform = transform
        path = path + ("train/" if train else "test/")
        self.pathX = path + "X/"
        self.pathY = path + "Y/"
        self.data = os.listdir(self.pathX)
        self.train = train

    def __getitem__(self, idx):
        ''' Load data and apply transformations'''

        f = self.data[idx]       

        img0 = cv2.imread(self.pathX + f + "/rgb/0.png")
        img1 = cv2.imread(self.pathX + f + "/rgb/1.png")
        img2 = cv2.imread(self.pathX + f + "/rgb/2.png")

        depth = np.load(self.pathX + f + "/depth.npy")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)  

        field_id = pickle.load(open(self.pathX + f + "/field_id.pkl", "rb"))  

        if self.train:
            Y = np.load(self.pathY + f + ".npy")
            return (img0, img1, img2, depth) , Y, field_id
        else:
            return (img0, img1, img2, depth), field_id
    
    def __len__(self):
        return len(self.data) 
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = os.path.dirname(os.path.realpath(__file__)) + "/lazydata/"
torch.backends.cudnn.benchmark = True

transforms1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

n_epochs = 30
momentum = 0.5
lr = 0.01              
loss_fn = nn.MSELoss() 
batch_size = 32

dataset_train = LazyLoadDataSet(path, transform = transforms1)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle=True) 

dataset_test = LazyLoadDataSet(path, train = False, transform = transforms1)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 1, shuffle = True)

# preview a single observation
preview = dataset_train[0]
print("Train images/depth shape: ", preview[0][0].shape) 
print("Train Y Label Shape: ",preview[1].shape)   
print("Observation ID (str): ",preview[2])       
for batch, (data, target, ids) in enumerate(loader_train):
    print("Batched images/depth shape: ", data[0].shape)
    break

# model settings / variables
input_size  = 224*224   
output_size = 12

# create model as prebuilt resnet with default trained weights from ImageNet
model = resnet18(weights = ResNet18_Weights.DEFAULT)
model.to(device)
model.fc = nn.Linear(in_features = 512, out_features = 12, bias = True)

optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum) 

sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', 
    factor = 0.5, patience = 2, threshold = 0.00001, cooldown = 1, 
    min_lr = 0, eps = 1e-5)

def train(epoch, model, optimizer, loss_fn):
    model.train()
    for batch_idx, (data, target, ids) in enumerate(loader_train):

        # make sure all vars have the same device as model
        data[0], target = data[0].to(device), target.to(device)

        # permute the pixels if set
        target = target.type(torch.float)

        # compute predict error
        output = model.forward(data[0]).type(torch.float) # model.forward(data) 
        loss = torch.sqrt(loss_fn(output, target))

        # back propagation
        optimizer.zero_grad() # erase all gradients before computing new one
        loss.backward()       # compute gradients of params w.r.t. loss (chain rule)
        optimizer.step()      # update weights using gradients and learning rate
    
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx * batch_size, 
                len(loader_train.dataset),
                100. * batch_idx / len(loader_train), 
                loss.item()))
           
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, 
        len(loader_train.dataset), 
        len(loader_train.dataset),
        100. * batch_idx / len(loader_train), 
        loss.item()))
    sched.step(loss)

for epoch in range(n_epochs):
    train(epoch, model, optimizer, loss_fn)

# output code here
outfile = 'submission.csv'
output_file = open(outfile, 'w')
titles = ['ID', 'FINGER_POS_1', 'FINGER_POS_2', 'FINGER_POS_3', 
          'FINGER_POS_4', 'FINGER_POS_5', 'FINGER_POS_6',
         'FINGER_POS_7', 'FINGER_POS_8', 'FINGER_POS_9', 
         'FINGER_POS_10', 'FINGER_POS_11', 'FINGER_POS_12']

field_ids = []
preds = []

model.eval()

for i, (data, field_id) in enumerate(loader_test):
    field_ids.append(field_id)
    pred = model(data[0])
    preds.append(pred.cpu().detach().numpy())

df = pd.concat([pd.DataFrame(field_ids), pd.DataFrame(np.concatenate(preds))], axis = 1, names = titles)
df.columns = titles
df.to_csv(outfile, index = False)
print("Written to csv file {}".format(outfile))

""" 
Preview of sample training cycle below:
Train Epoch: 1 [2200/3396 (65%)]	Loss: 0.001941
"""