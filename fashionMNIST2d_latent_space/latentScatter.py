#import sympy as sp
#import minterpy as mp
import numpy as np
#from minterpy.pointcloud_utils import *

#from mpl_toolkits import mplot3d
#%matplotlib inline

#from mpl_toolkits.mplot3d import axes3d

import torch
import torchvision
from torchvision import transforms, datasets

#import random
import numpy as np
import matplotlib.pyplot as plt

import os
from operator import itemgetter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#import ot

#from sklearn.neighbors import NearestNeighbors

#import ripser
#import persim
#from persim import plot_diagrams

from operator import itemgetter


#do pip installs as follows to use vedo for plotting point clouds
#pip install vedo
#pip install ipyvtklink

import numpy as np
#from vedo import *

import matplotlib.pyplot
#from mpl_toolkits.mplot3d import Axes3D



def getFashionMNIST(batch_size = 1, drop_last=False):
    fashionmnist_data = torchvision.datasets.FashionMNIST(download=True, root = 'data/fashionmnist', transform = 
                                                                                 transforms.Compose([transforms.Resize(32),
                                                                                 transforms.ToTensor(), 
                                                                                 transforms.Lambda(lambda x: x.repeat(1, 1, 1))
                                                                                 ]))

    fashionmnist_data_test = torchvision.datasets.FashionMNIST(download=True, root = 'data/fashionmnist', train=False, transform = 
                                                                                 transforms.Compose([transforms.Resize(32),
                                                                                 transforms.ToTensor(), 
                                                                                 transforms.Lambda(lambda x: x.repeat(1, 1, 1))
                                                                                 ]))

    train_loader = torch.utils.data.DataLoader(fashionmnist_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              drop_last=drop_last)

    test_loader = torch.utils.data.DataLoader(fashionmnist_data_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              drop_last=drop_last)

    return train_loader, test_loader



def getDataset(dataset = "MNIST", batch_size = 1):
    if(dataset == "MNIST"):
        train_loader, test_loader = getMNIST(batch_size)
        noChannels,dx, dy = train_loader.dataset.__getitem__(1)[0].shape
    elif(dataset == "FashionMNIST"):
        train_loader, test_loader = getFashionMNIST(batch_size)
        noChannels, dx, dy = train_loader.dataset.__getitem__(1)[0].shape
    elif(dataset == "Cifar10"):
        train_loader, test_loader = getCifar10(batch_size)
        noChannels, dx, dy = train_loader.dataset.__getitem__(1)[0].shape
    else:
        return None, None, None, None, None    
        
    return train_loader, test_loader, noChannels, dx, dy


train_loader, test_loader, noChannels, dx, dy = getDataset("FashionMNIST")  # FashionMNIST , MNIST


label0 = torch.tensor([])
label1 = torch.tensor([])
label2 = torch.tensor([])
label3 = torch.tensor([])
label4 = torch.tensor([])
label5 = torch.tensor([])
label6 = torch.tensor([])
label7 = torch.tensor([])
label8 = torch.tensor([])
label9 = torch.tensor([])

for inum, (batch_x, label) in enumerate(test_loader):
    
    if(label==0):
        label0 = torch.cat((label0, batch_x))
    if(label==1):
        label1 = torch.cat((label1, batch_x))
    if(label==2):
        label2 = torch.cat((label2, batch_x))
    if(label==3):
        label3 = torch.cat((label3, batch_x))
    if(label==4):
        label4 = torch.cat((label4, batch_x))
    if(label==5):
        label5 = torch.cat((label5, batch_x))
    if(label==6):
        label6 = torch.cat((label6, batch_x))
    if(label==7):
        label7 = torch.cat((label7, batch_x))
    if(label==8):
        label8 = torch.cat((label8, batch_x))
    if(label==9):
        label9 = torch.cat((label9, batch_x))


torch.save(label0, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label0.pt')
torch.save(label1, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label1.pt')
torch.save(label2, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label2.pt')
torch.save(label3, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label3.pt')
torch.save(label4, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label4.pt')
torch.save(label5, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label5.pt')
torch.save(label6, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label6.pt')
torch.save(label7, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label7.pt')
torch.save(label8, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label8.pt')
torch.save(label9, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label9.pt')

check = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label9.pt')

print('check.shape', check.shape)