import sys
sys.path.append('/home/ramana44/autoencoder-regularisation-')

from get_data import get_data
import torch
import os
import numpy as np

from datasets import InMemDataLoader

from train_ae_LSTSQ20_noSamples100degPoly55 import train
from activations import Sin
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import wandb
# wandb.login()

reg_nodes_sampling = "legendre"
alpha = 0.5
hidden_size = 100
deg_poly = 21
latent_dim = 2
lr = 1e-4
no_layers = 3
train_set_size = 0.1


batch_size_cfs = 200
HybridPolyDegree = 20

coeffs_saved_trn = torch.load('/home/ramana44/withR_matrix_Fmnist_RK_method/coeffs_saved/LSTSQ_traincoeffs_FMNIST_dq'+str(HybridPolyDegree)+'.pt').to(device)
image_batches_trn = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/trainImages.pt').to(device)
image_batches_trn = image_batches_trn.reshape(int(image_batches_trn.shape[0]/batch_size_cfs), batch_size_cfs, 1, 32,32)
coeffs_saved_trn = coeffs_saved_trn.reshape(int(coeffs_saved_trn.shape[0]/batch_size_cfs), batch_size_cfs, coeffs_saved_trn.shape[1]).unsqueeze(2) 

coeffs_saved_test = torch.load('/home/ramana44/withR_matrix_Fmnist_RK_method/coeffs_saved/LSTSQ_testcoeffs_FMNIST_dq'+str(HybridPolyDegree)+'.pt').to(device)
image_batches_test = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/testImages.pt').to(device)
image_batches_test = image_batches_test.reshape(int(image_batches_test.shape[0]/batch_size_cfs), batch_size_cfs, 1, 32,32)
coeffs_saved_test = coeffs_saved_test.reshape(int(coeffs_saved_test.shape[0]/batch_size_cfs), batch_size_cfs, coeffs_saved_test.shape[1]).unsqueeze(2) 


model, model_reg, loss_arr_reg, loss_arr_reco, loss_arr_base, loss_arr_val_reco, loss_arr_val_base = train(image_batches_trn, image_batches_test, coeffs_saved_trn, coeffs_saved_test, no_epochs=100, reco_loss="mse", latent_dim=latent_dim, 
        hidden_size=hidden_size, no_layers=no_layers, activation = Sin(), lr=lr, alpha = alpha, bl=False,
        seed = 2342, train_base_model=True, no_samples=5, deg_poly=deg_poly,
        reg_nodes_sampling=reg_nodes_sampling, no_val_samples = 10, HybridPolyDegree = 20, use_guidance = False, train_set_size=train_set_size,
        enable_wandb=False, wandb_project='Test_mrt', wandb_entity='ae_reg_team')







