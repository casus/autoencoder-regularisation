import sys
sys.path.append('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders')

from datasets import  getDataset
import wandb

import os
from models import AE
import torch

train_loader, test_loader, noChannels, dx, dy = getDataset('FashionMNIST', 200, False)

import circle_MLP_AEREG_train_def as train_ae
from activations import Sin

(model, model_reg, loss_arr_reg, loss_arr_reco, loss_arr_base, loss_arr_val_reco, loss_arr_val_base) = train_ae.train(train_loader, test_loader, noChannels, dx, dy, no_epochs=40, TDA=1.0, reco_loss="mse", latent_dim=2, 
          hidden_size=6, no_layers=2, activation = Sin(), lr=0.002, alpha = 0.5,
          seed = 2342, train_base_model=True, no_samples=10, deg_poly=21,
          reg_nodes_sampling='legendre', no_val_samples = 10, use_guidance = False,
          enable_wandb=False, wandb_project='ae_reg', wandb_entity='ae_reg_team')


