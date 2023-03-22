import sys
#sys.path.append('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders')


sys.path.append('./')


#from get_data import get_data, get_data_train, get_data_val
import torch
import os
import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline

from datasets import InMemDataLoader
import torch.nn.functional as F
import torch
import nibabel as nib     # Read / write access to some common neuroimaging file formats
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
from scipy import interpolate
import ot

import jmp_solver1.surrogates
import matplotlib
matplotlib.rcdefaults() 

path_in_repo = './models_saved/'

deg_quad = 20

# load trained rAE and bAE

#from models_un import AE_un
from models import AE
from activations import Sin

path_hyb = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_space/output/MRT_full/test_run_saving/'
path_unhyb = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'

#specify hyperparameters
reg_nodes_sampling = 'legendre'
alpha = 0.5
frac = 0.4
hidden_size = 100
deg_poly = 21
deg_poly_forRK = 21
latent_dim = 2
lr = 0.0001
no_layers = 3
no_epochs= 100
name_hyb = '_'+reg_nodes_sampling+'_'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly_forRK)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)
name_unhyb = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

#no_channels, dx, dy = (train_loader_alz.dataset.__getitem__(1).shape)
#inp_dim = [no_channels, dx-21, dy-21]
inp_dim_hyb = (deg_quad+1)*(deg_quad+1)

inp_dim_unhyb = [1,32,32]

RK_model_reg = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
RK_model_base = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

model_reg = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
model_base = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

#model_reg.load_state_dict(torch.load(path+'model_reg'+name, map_location=torch.device('cpu'))["model"])
#model_base.load_state_dict(torch.load(path+'model_reg'+name, map_location=torch.device('cpu'))["model"])

#path_hyb = path_in_repo
#path_unhyb = path_in_repo

RK_model_reg.load_state_dict(torch.load(path_hyb+'model_regLSTQS'+str(deg_quad)+''+name_hyb, map_location=torch.device('cpu')))
RK_model_base.load_state_dict(torch.load(path_hyb+'model_baseLSTQS'+str(deg_quad)+''+name_hyb, map_location=torch.device('cpu')))

torch.save(RK_model_reg.state_dict(), path_in_repo+'model_regLSTQS'+str(deg_quad)+''+name_hyb)
torch.save(RK_model_base.state_dict(), path_in_repo+'model_baseLSTQS'+str(deg_quad)+''+name_hyb)

model_reg.load_state_dict(torch.load(path_unhyb+'model_reg_TDA'+name_unhyb, map_location=torch.device('cpu')))
model_base.load_state_dict(torch.load(path_unhyb+'model_base_TDA'+name_unhyb, map_location=torch.device('cpu')))

torch.save(model_reg.state_dict(), path_in_repo+'/model_reg_TDA'+name_unhyb)
torch.save(model_base.state_dict(), path_in_repo+'/model_base_TDA'+name_unhyb)

#model_reg.eval()
#model_base.eval()

print("Anything here to see")


#loading convolutional autoencoder
from convAE import ConvoAE
no_layers_cae = 3
latent_dim_cae = latent_dim
lr_cae =1e-3
name_unhyb_cae = '_'+str(frac)+'_'+str(latent_dim_cae)+'_'+str(lr_cae)+'_'+str(no_layers_cae)
model_convAE = ConvoAE(latent_dim_cae).to(device)
model_convAE.load_state_dict(torch.load(path_unhyb+'model_base_cae_TDA'+name_unhyb_cae, map_location=torch.device(device)), strict=False)
torch.save(model_convAE.state_dict(), path_in_repo+'/model_base_cae_TDA'+name_unhyb_cae)


#rec = model_convAE(trainImages).view(trainImages.shape).detach().numpy() 

#from vae import BetaVAE
from activations import Sin
activation = Sin()
from vae_models_for_fmnist import VAE_try, VAE_mlp, Autoencoder_linear
#model_mlpVAE = MLPVAE(1*32*32, hidden_size, latent_dim, 
                #    no_layers, activation).to(device) # regularised autoencoder

model_mlpVAE_ = VAE_mlp(32*32, hidden_size, latent_dim).to(device)

#model_betaVAE = BetaVAE([32, 32], 1, no_filters=4, no_layers=3,
                #kernel_size=3, latent_dim=10, activation = Sin()).to(device) # regularised autoencoder

model_cnnVAE_ = VAE_try(image_channels=1, h_dim=8*2*2, z_dim=latent_dim).to(device)

#model_betaVAE = BetaVAE(batch_size = 1, img_depth = 1, net_depth = no_layers, z_dim = latent_dim, img_dim = 32).to(device)
model_mlpVAE_.load_state_dict(torch.load(path_unhyb+'model_base_mlp_vae_TDA'+name_unhyb, map_location=torch.device(device)), strict=False)
model_cnnVAE_.load_state_dict(torch.load(path_unhyb+'model_base_cnn_vae_TDA'+name_unhyb, map_location=torch.device(device)), strict=False)
torch.save(model_cnnVAE_.state_dict(), path_in_repo+'/model_base_cnn_vae_TDA'+name_unhyb)
torch.save(model_mlpVAE_.state_dict(), path_in_repo+'/model_base_mlp_vae_TDA'+name_unhyb)




def model_mlpVAE(input):
    #print('model_betaVAE(input).shape', model_betaVAE(input).shape)
    input = input.reshape(-1, 32*32)
    recon,_,_= model_mlpVAE_(input)
    return recon

def model_cnnVAE(input):
    #print('model_betaVAE(input).shape', model_betaVAE(input).shape)
    recon,_,_= model_cnnVAE_(input)
    return recon

def model_contra(input):
    input = input.reshape(-1, 32*32)
    recon,_,_= model_cnnVAE_(input)
    return recon

#rec = model_convAE(torch.from_numpy(trainImages).reshape(1,1,32,32).to(device))

#loading contractive autoencoder
no_layers_contraae = 3
latent_dim_contraae = latent_dim
lr_contraae =1e-3
name_unhyb_contraae = '_'+str(frac)+'_'+str(latent_dim_contraae)+'_'+str(lr_contraae)+'_'+str(no_layers_contraae)
model_contra_ = Autoencoder_linear(latent_dim).to(device)
model_contra_.load_state_dict(torch.load(path_unhyb+'model_base_contraAE_TDA'+name_unhyb_contraae, map_location=torch.device(device)), strict=False)
torch.save(model_contra_.state_dict(), path_in_repo+'/model_base_contraAE_TDA'+name_unhyb_contraae)


def model_contra(input):
    input = input.reshape(-1, 32*32)
    recon= model_contra_(input)
    return recon

print("All loaded and saved ?")