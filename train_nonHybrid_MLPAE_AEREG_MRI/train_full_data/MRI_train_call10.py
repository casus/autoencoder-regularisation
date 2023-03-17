import sys
sys.path.append('./')

from datasets import  getDataset
import wandb
from datasets import InMemDataLoader
from get_data import get_data
import numpy as np

import os
from models import AE
import torch

#Directly loading the dataset presaved as torch tensor after randoming the samples
# The dataset consists of 50000 images in the training set and 10000 images in the test set

# Data presaved to manualy experiment with different amounts of training data

trainImagesMRI = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/trainDataSet.pt',map_location=torch.device('cuda'))
#shape: 60000, 1, 96, 96

testImagesMRI = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/testDataSet.pt',map_location=torch.device('cuda'))
testImagesMRI = testImagesMRI[:10000]
#shape: 10000, 1, 96, 96

noChannels = trainImagesMRI.shape[1] # single channel
dx =trainImagesMRI.shape[2]   # 96 is the image size along x direction
dy = trainImagesMRI.shape[3]  # 96 is the image size along y direction

# Setting up the dataset in batches

batch_size = 200
trainImagesInBatches = trainImagesMRI.reshape(int(trainImagesMRI.shape[0]/batch_size), batch_size, 1, 96,96)
testImagesInBatches = testImagesMRI.reshape(int(testImagesMRI.shape[0]/batch_size), batch_size, 1, 96,96)

import train_ae_MLPAE_AEREG_MRI as train_ae
from activations import Sin


all_paths = []
for root, dirs, files in os.walk(os.path.abspath("/home/ramana44/all_scans_single_channel_equal_dim/")):
    for file in files:
        #print(os.path.join(root, file))
        all_paths.append((os.path.join(root, file)))

#print('len(all_paths)',len(all_paths))


### load data ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_train_test_set(paths, device, batch_size=200, train_set_size=0.8, test_set_size=0.2):
    #assert train_set_size + test_set_size <= 1., "Train and test set size should not exceed 100%"
    
    path_indices = np.arange(len(paths))
    #np.random.shuffle(path_indices)                             # randomize indices of the paths for train and test set selection
    
    num_train = int(np.round_(len(paths) * train_set_size))     # calc amount of training sets to load
    num_test = int(np.round_(len(paths) * test_set_size))       # calc amount of test sets to load
    train_indices = path_indices[:num_train]                    # select unique and random indices from all paths
    test_indices = path_indices[-num_test:]                     # for train and test set


    train_data = get_data([paths[i] for i in train_indices], device)  # only load specific indices preveiously selected
    test_data = get_data([paths[i] for i in test_indices], device)
    #print('train_data.shape',train_data.shape)

    train_loader = InMemDataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) # init dataloader for train and test set
    test_loader = InMemDataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True) 
    return train_loader, test_loader

train_loader, test_loader = get_train_test_set(all_paths, device, train_set_size=0.1, batch_size=200)


(model, model_reg, loss_arr_reg, loss_arr_reco, loss_arr_base, loss_arr_val_reco, loss_arr_val_base) = train_ae.train(train_loader, test_loader, noChannels, dx, dy, no_epochs=100, TDA=0.8, reco_loss="mse", latent_dim=10, 
          hidden_size=1000, no_layers=5, activation = Sin(), lr=0.0001, alpha = 0.5,
          seed = 2342, train_base_model=True, no_samples=10, deg_poly=21,
          reg_nodes_sampling='legendre', no_val_samples = 10, use_guidance = False,
          enable_wandb=False, wandb_project='ae_reg', wandb_entity='ae_reg_team', weight_jac = False)