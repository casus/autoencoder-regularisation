import sys
sys.path.append('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders')

from datasets import  getDataset
import wandb

import os
from models import AE
import torch

train_loader, test_loader, noChannels, dx, dy = getDataset('FashionMNIST', 200, False)

import train_ae_MLPAE_AEREG_MRI as train_ae
from activations import Sin

(model, model_reg, loss_arr_reg, loss_arr_reco, loss_arr_base, loss_arr_val_reco, loss_arr_val_base) = train_ae.train(train_loader, test_loader, noChannels, dx, dy, no_epochs=100, TDA=0.05, reco_loss="mse", latent_dim=10, 
          hidden_size=1000, no_layers=5, activation = Sin(), lr=0.0001, alpha = 0.5,
          seed = 2342, train_base_model=True, no_samples=10, deg_poly=21,
          reg_nodes_sampling='legendre', no_val_samples = 10, use_guidance = False,
          enable_wandb=False, wandb_project='ae_reg', wandb_entity='ae_reg_team')

