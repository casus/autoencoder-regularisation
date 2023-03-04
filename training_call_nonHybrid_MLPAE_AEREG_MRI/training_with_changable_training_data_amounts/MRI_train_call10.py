import sys
#sys.path.append('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders')
sys.path.append('./autoencoder-regularisation-')

from datasets import  getDataset
import wandb

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


(model, model_reg, loss_arr_reg, loss_arr_reco, loss_arr_base, loss_arr_val_reco, loss_arr_val_base) = train_ae.train(trainImagesInBatches, testImagesInBatches, noChannels, dx, dy, no_epochs=100, TDA=0.8, reco_loss="mse", latent_dim=10, 
          hidden_size=1000, no_layers=5, activation = Sin(), lr=0.0001, alpha = 0.5,
          seed = 2342, train_base_model=True, no_samples=10, deg_poly=21,
          reg_nodes_sampling='legendre', no_val_samples = 10, use_guidance = False,
          enable_wandb=False, wandb_project='ae_reg', wandb_entity='ae_reg_team', weight_jac = False)