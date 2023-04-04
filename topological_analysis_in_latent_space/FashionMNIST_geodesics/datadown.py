# This program doiwnloads and saves the FashionMNIST dataset

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import os

# Download the dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


# Load the test data set as torch tensors
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)

# print the test dataset as torch tensors

# Save the test dataset as numpy arrays
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs = inputs.numpy()
    labels = labels.numpy()
    print("inputs.shape", inputs.shape)
    inputs = torch.tensor(inputs)
