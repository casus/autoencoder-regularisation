import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from models_of_VAEs import BetaVAE, MLPVAE
from vae_models_for_fmnist import VAE_try, VAE_mlp
import numpy as np
import matplotlib.pyplot as plt

import os
import re
from datasets import getMNIST, getFashionMNIST, getCifar10, getDataset
import copy

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cpu')
device = torch.device('cuda')
from models import AE
#from vae import BetaVAE
from activations import Sin
from regularisers import computeC1Loss
#from models_circle import MLPVAE

#from tabulate import tabulate

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

filename = "table_model_reg_legendre_legendre.txt"
latent_dim = 10


class Autoencoder_linear(nn.Module):
    def __init__(self):

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32*32, 100),  #input layer
            nn.ReLU(),
            nn.Linear(100, 100),   #h1
            nn.ReLU(),
            nn.Linear(100, 100),    #h1
            nn.ReLU(),
            nn.Linear(100, 100),    #h1
            nn.ReLU(),
            nn.Linear(100,latent_dim)  # latent layer
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 100),  #input layer
            nn.ReLU(),
            nn.Linear(100, 100),   #h1
            nn.ReLU(),
            nn.Linear(100, 100),    #h1
            nn.ReLU(),
            nn.Linear(100, 100),    #h1
            nn.ReLU(),
            nn.Linear(100, 32*32),  # latent layer
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#row arrangements
ori_row_im_ind = 0
mlp_ae_im_index = 1
convAE_im_index = 2
mlpVAE_img_ind = 3
betaVAE_img_ind = 4
contraAE_img_ind = 5
ae_reg_im_ind = 6
hyb_AEREG_inm_ind = 7



loadableDatas = ["train", "test"]
choosenData = loadableDatas[1]
#availableModels = ["baseline", "regularized"]
#modelSelected = availableModels[1]

coeff_sol_method = "LSTQS"

ChoosenImageIndex = 13

#Hybrid AE-REG parameters
Hybrid_poly_deg = 16


# MLPAE and AE-REG parameters
hidden_size = 100
alpha = 0.5
frac = 0.4

prozs = [0.1, 0.2, 0.5, 0.7, ]

rand_perturb = []
orig_perturb = []
rec_perturb = []

path_ = '/home/ramana44/autoencoder_regulrization_conf_tasks/models/'
paths = [path_+'model_base_legendre_', path_+'model_reg_trainingData_', path_+'model_reg_legendre_', '/home/willma32/regularizedautoencoder/output/FMNIST_vae/model_reg_']

names = ['baseline', 'contractive', 'legendre', 'vae']

flh = transforms.RandomHorizontalFlip(p=1.)
flv = transforms.RandomVerticalFlip(p=1.)

labels = ['baseline', 'contractive', 'legendre', 'vae']
path_file = '/home/ramana44/autoencoder_regulrization_conf_tasks/FMNIST_samples/'
path_to_dir = '/home/ramana44/autoencoder_regulrization_conf_tasks_hybrid/output_FMNIST_trans/all_AEs/image_'+str(ChoosenImageIndex)+'/lstqs_deg_'+str(Hybrid_poly_deg)+'_all_lats/'

global_ind = 0

rand_perturb = []
orig_perturb = []
rec_perturb = []
path_to_model = paths[2] + str(alpha)+'_'+str(latent_dim)+'_'+str(hidden_size)+'_'+str(frac)


# loading MLPAE and AE-REG
from models import AE
from activations import Sin
path_hyb = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_space/output/MRT_full/test_run_saving/'
path_unhyb = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'

#specify hyperparameters
reg_nodes_sampling = 'legendre'

deg_poly = 21
lr = 0.0001
no_layers = 3
no_epochs= 100

name_unhyb = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

#name_unhyb = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)


inp_dim_hyb = (Hybrid_poly_deg+1)*(Hybrid_poly_deg+1)

inp_dim_unhyb = [1,32,32]

model_reg = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
model_base = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

model_reg.load_state_dict(torch.load(path_unhyb+'model_reg_TDA'+name_unhyb, map_location=torch.device(device)))
model_base.load_state_dict(torch.load(path_unhyb+'model_base_TDA'+name_unhyb, map_location=torch.device(device)))


#loading convolutional autoencoder
from convAE import ConvoAE
no_layers_cae = 3
latent_dim_cae = latent_dim
lr_cae =1e-3
name_unhyb_cae = '_'+str(frac)+'_'+str(latent_dim_cae)+'_'+str(lr_cae)+'_'+str(no_layers_cae)
model_convAE = ConvoAE(latent_dim_cae).to(device)
model_convAE.load_state_dict(torch.load(path_unhyb+'model_base_cae_TDA'+name_unhyb_cae, map_location=torch.device(device)), strict=False)


#from vae import BetaVAE
from activations import Sin
activation = Sin()
#model_mlpVAE = MLPVAE(1*32*32, hidden_size, latent_dim, 
                #    no_layers, activation).to(device) # regularised autoencoder

model_mlpVAE = VAE_mlp(32*32, hidden_size, latent_dim).to(device)

#model_betaVAE = BetaVAE([32, 32], 1, no_filters=4, no_layers=3,
                #kernel_size=3, latent_dim=10, activation = Sin()).to(device) # regularised autoencoder

model_betaVAE = VAE_try(image_channels=1, h_dim=8*2*2, z_dim=latent_dim).to(device)

#model_betaVAE = BetaVAE(batch_size = 1, img_depth = 1, net_depth = no_layers, z_dim = latent_dim, img_dim = 32).to(device)
model_mlpVAE.load_state_dict(torch.load(path_unhyb+'model_base_mlp_vae_TDA'+name_unhyb, map_location=torch.device(device)), strict=False)
model_betaVAE.load_state_dict(torch.load(path_unhyb+'model_base_cnn_vae_TDA'+name_unhyb, map_location=torch.device(device)), strict=False)


#loading contractive autoencoder
no_layers_contraae = 3
latent_dim_contraae = latent_dim
lr_contraae =1e-3
name_unhyb_contraae = '_'+str(frac)+'_'+str(latent_dim_contraae)+'_'+str(lr_contraae)+'_'+str(no_layers_contraae)
model_contra = Autoencoder_linear().to(device)
model_contra.load_state_dict(torch.load(path_unhyb+'model_base_contraAE_TDA'+name_unhyb_contraae, map_location=torch.device(device)), strict=False)


model = model_base
orig = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images.pt')
orig = np.array(orig[ChoosenImageIndex])
print(orig.shape)

orig = orig.reshape(1,1024)

orig_flh = flh(torch.from_numpy(orig).reshape(1,32,32).to(device))
orig_flv = flv(torch.from_numpy(orig).reshape(1,32,32).to(device))

for proz in prozs:
    rand_perturb.append(np.random.rand(1, 1024)*(np.max(orig)-np.min(orig))*proz)
print((rand_perturb[0].shape))

rec_flv = model(flv(torch.from_numpy(orig).reshape(1,32,32).to(device)))
rec_flh = model(flh(torch.from_numpy(orig).reshape(1,32,32).to(device)))

for rand_transform in rand_perturb:
    orig_perturb.append(torch.from_numpy(np.add(orig,rand_transform)).reshape(1,32,32).to(device))
    rec_perturb.append(model(orig_perturb[-1].float()))


fig = plt.figure(constrained_layout=True)
fig, ax = plt.subplots(8,3+len(rand_perturb), figsize=((3+len(rand_perturb))*3, 3*3))
ax[ori_row_im_ind,0].imshow(orig.reshape(32,32))

ax[ori_row_im_ind,1].imshow(orig_flh.detach().cpu().numpy().reshape(32,32))
ax[ori_row_im_ind,2].imshow(orig_flv.detach().cpu().numpy().reshape(32,32))

print("len(rand_perturb)",len(rand_perturb))
for k in range(len(rand_perturb)):
    ax[ori_row_im_ind, 3+k].imshow(orig_perturb[k].detach().cpu().numpy().reshape(32,32)) 


rec = model(torch.from_numpy(orig).reshape(1,32,32).to(device))
ax[mlp_ae_im_index,0].imshow(rec.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec.detach().cpu().numpy().reshape(32,32)), np.max(orig.reshape(32,32)))
minv = min(np.min(rec.detach().cpu().numpy().reshape(32,32)), np.min(orig.reshape(32,32)))
rec_n = (rec.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig.reshape(32,32) - minv)/(maxv-minv)*255.

ax[mlp_ae_im_index,1].imshow(rec_flh.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flh.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flh.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

ax[mlp_ae_im_index,2].imshow(rec_flv.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flv.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flv.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

for k in range(len(rand_perturb)):
    print('len(rand_perturb)', len(rand_perturb))
    print('k',k)
    ax[mlp_ae_im_index,3+k].imshow(rec_perturb[k].detach().cpu().numpy().reshape(32,32))
    maxv = max(np.max(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.max(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    minv = min(np.min(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.min(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    rec_n = (rec_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
    orig_n = (orig_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
 
####################################################################################################################
# MLP VAE start

rand_perturb = []
orig_perturb = []
rec_perturb = []
#model = model_betaVAE

def model(input):
    #print('model_betaVAE(input).shape', model_betaVAE(input).shape)
    input = input.reshape(-1, 32*32)
    recon,_,_= model_mlpVAE(input)
    return recon

#model = vae_model()

orig = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images.pt')
orig = np.array(orig[ChoosenImageIndex])
print(orig.shape)


orig = orig.reshape(1,1024)

orig_flh = flh(torch.from_numpy(orig).reshape(1,32,32).to(device))
orig_flv = flv(torch.from_numpy(orig).reshape(1,32,32).to(device))

for proz in prozs:
    rand_perturb.append(np.random.rand(1, 1024)*(np.max(orig)-np.min(orig))*proz)
print((rand_perturb[0].shape))

rec_flv = model(torch.tensor(flv(torch.from_numpy(orig).reshape(1,1,32,32))).to(device))
rec_flh = model(flh(torch.from_numpy(orig).reshape(1,1,32,32).to(device)))

for rand_transform in rand_perturb:
    orig_perturb.append(torch.from_numpy(np.add(orig,rand_transform)).reshape(1,1,32,32).to(device))
    rec_perturb.append(model(orig_perturb[-1].float()))


rec = model(torch.from_numpy(orig).reshape(1,1,32,32).to(device))
print('what happened rec.shape', rec)
ax[mlpVAE_img_ind,0].imshow(rec.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec.detach().cpu().numpy().reshape(32,32)), np.max(orig.reshape(32,32)))
minv = min(np.min(rec.detach().cpu().numpy().reshape(32,32)), np.min(orig.reshape(32,32)))
rec_n = (rec.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig.reshape(32,32) - minv)/(maxv-minv)*255.

ax[mlpVAE_img_ind,1].imshow(rec_flh.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flh.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flh.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

ax[mlpVAE_img_ind,2].imshow(rec_flv.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flv.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flv.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

for k in range(len(rand_perturb)):
    print("k", k)
    print('len(rand_perturb)', len(rand_perturb))
    ax[mlpVAE_img_ind,3+k].imshow(rec_perturb[k].detach().cpu().numpy().reshape(32,32))
    maxv = max(np.max(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.max(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    minv = min(np.min(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.min(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    rec_n = (rec_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
    orig_n = (orig_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

#####################################################################################################################


####################################################################################################################
# CNN VAE start

rand_perturb = []
orig_perturb = []
rec_perturb = []
#model = model_betaVAE

def model(input):
    #print('model_betaVAE(input).shape', model_betaVAE(input).shape)
    recon,_,_= model_betaVAE(input)
    return recon

#model = vae_model()

orig = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images.pt')
orig = np.array(orig[ChoosenImageIndex])
print(orig.shape)


orig = orig.reshape(1,1024)

orig_flh = flh(torch.from_numpy(orig).reshape(1,32,32).to(device))
orig_flv = flv(torch.from_numpy(orig).reshape(1,32,32).to(device))

for proz in prozs:
    rand_perturb.append(np.random.rand(1, 1024)*(np.max(orig)-np.min(orig))*proz)
print((rand_perturb[0].shape))

rec_flv = model(torch.tensor(flv(torch.from_numpy(orig).reshape(1,1,32,32))).to(device))
rec_flh = model(flh(torch.from_numpy(orig).reshape(1,1,32,32).to(device)))

for rand_transform in rand_perturb:
    orig_perturb.append(torch.from_numpy(np.add(orig,rand_transform)).reshape(1,1,32,32).to(device))
    rec_perturb.append(model(orig_perturb[-1].float()))


rec = model(torch.from_numpy(orig).reshape(1,1,32,32).to(device))
print('what happened rec.shape', rec)
ax[betaVAE_img_ind,0].imshow(rec.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec.detach().cpu().numpy().reshape(32,32)), np.max(orig.reshape(32,32)))
minv = min(np.min(rec.detach().cpu().numpy().reshape(32,32)), np.min(orig.reshape(32,32)))
rec_n = (rec.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig.reshape(32,32) - minv)/(maxv-minv)*255.

ax[betaVAE_img_ind,1].imshow(rec_flh.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flh.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flh.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

ax[betaVAE_img_ind,2].imshow(rec_flv.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flv.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flv.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

for k in range(len(rand_perturb)):
    print("k", k)
    print('len(rand_perturb)', len(rand_perturb))
    ax[betaVAE_img_ind,3+k].imshow(rec_perturb[k].detach().cpu().numpy().reshape(32,32))
    maxv = max(np.max(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.max(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    minv = min(np.min(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.min(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    rec_n = (rec_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
    orig_n = (orig_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

#####################################################################################################################


#Contractive_AE

rand_perturb = []
orig_perturb = []
rec_perturb = []
model = model_contra

orig = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images.pt')
orig = np.array(orig[ChoosenImageIndex])
print(orig.shape)


orig = orig.reshape(1,1024)

orig_flh = flh(torch.from_numpy(orig).reshape(1,32,32).to(device))
orig_flv = flv(torch.from_numpy(orig).reshape(1,32,32).to(device))

for proz in prozs:
    rand_perturb.append(np.random.rand(1, 1024)*(np.max(orig)-np.min(orig))*proz)
print((rand_perturb[0].shape))

rec_flv = model(torch.tensor(flv(torch.from_numpy(orig))).to(device)).reshape(1,32,32)
rec_flh = model(flh(torch.from_numpy(orig).to(device))).reshape(1,32,32)

for rand_transform in rand_perturb:
    orig_perturb.append(torch.from_numpy(np.add(orig,rand_transform)).to(device))
    rec_perturb.append(model(orig_perturb[-1].float()))


rec = model(torch.from_numpy(orig).to(device)).reshape(1,1,32,32)
ax[contraAE_img_ind,0].imshow(rec.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec.detach().cpu().numpy().reshape(32,32)), np.max(orig.reshape(32,32)))
minv = min(np.min(rec.detach().cpu().numpy().reshape(32,32)), np.min(orig.reshape(32,32)))
rec_n = (rec.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig.reshape(32,32) - minv)/(maxv-minv)*255.

ax[contraAE_img_ind,1].imshow(rec_flh.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flh.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flh.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

ax[contraAE_img_ind,2].imshow(rec_flv.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flv.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flv.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

for k in range(len(rand_perturb)):
    print("k", k)
    print('len(rand_perturb)', len(rand_perturb))
    ax[contraAE_img_ind,3+k].imshow(rec_perturb[k].detach().cpu().numpy().reshape(32,32))
    maxv = max(np.max(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.max(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    minv = min(np.min(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.min(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    rec_n = (rec_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
    orig_n = (orig_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.


###########################################################################################################################

orig_perturb = []
rec_perturb = []

model = model_reg

rec_flv = model(flv(torch.from_numpy(orig).reshape(1,32,32).to(device)))
rec_flh = model(flh(torch.from_numpy(orig).reshape(1,32,32).to(device)))

for rand_transform in rand_perturb:
    orig_perturb.append(torch.from_numpy(np.add(orig,rand_transform)).reshape(1,32,32).to(device))
    rec_perturb.append(model(orig_perturb[-1].float()))


rec = model(torch.from_numpy(orig).reshape(1,32,32).to(device))
ax[ae_reg_im_ind,0].imshow(rec.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec.detach().cpu().numpy().reshape(32,32)), np.max(orig.reshape(32,32)))
minv = min(np.min(rec.detach().cpu().numpy().reshape(32,32)), np.min(orig.reshape(32,32)))
rec_n = (rec.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig.reshape(32,32) - minv)/(maxv-minv)*255.

ax[ae_reg_im_ind,1].imshow(rec_flh.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flh.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flh.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

ax[ae_reg_im_ind,2].imshow(rec_flv.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flv.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flv.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

for k in range(len(rand_perturb)):
    ax[ae_reg_im_ind,3+k].imshow(rec_perturb[k].detach().cpu().numpy().reshape(32,32))
    maxv = max(np.max(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.max(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    minv = min(np.min(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.min(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    rec_n = (rec_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
    orig_n = (orig_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.


#####################################################################################################################
from matplotlib.style import available
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

import os
import re
from datasets import getMNIST, getFashionMNIST, getCifar10, getDataset
import copy

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
from models import AE
from vae import BetaVAE
from activations import Sin
from regularisers import computeC1Loss
from models_circle import MLPVAE

#from tabulate import tabulate

#from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import peak_signal_noise_ratio as psnr


#imports for Runge kutta
import scipy
import scipy.integrate
from jmp_solver1.sobolev import Sobolev
from jmp_solver1.sobolev import Sobolev
from jmp_solver1.solver import Solver
from jmp_solver1.utils import matmul
import jmp_solver1.surrogates
import time
#import minterpy as mp
from jmp_solver1.diffeomorphisms import hyper_rect
#imports for Runge Kutta done



#loadableDatas = ["train", "test"]
#choosenData = loadableDatas[1]
#availableModels = ["RK_baseline", "RK_regularized"]
#modelSelected = availableModels[0]

#ChoosenImageIndex = 9
#RK_quadratureDegree = 25
deg_quad = Hybrid_poly_deg

#hidden_size = 100

alphas = [0.1]
#alpha = 0.1

latent_dims = [2,3,4,5,6,7,8,9,10]
#latent_dim = 10

#fracs = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
#frac = 0.01

prozs = [0.1, 0.2, 0.5, 0.7, ]

rand_perturb = []
orig_perturb = []
rec_perturb = []
#model = AE([1,32,32], hidden_size, latent_dim, 3, Sin()).to('cuda')
path_ = '/home/ramana44/autoencoder_regulrization_conf_tasks_hybrid/models/'
paths = [path_+'model_base_legendre_', path_+'model_reg_trainingData_', path_+'model_reg_legendre_', '/home/willma32/regularizedautoencoder/output/FMNIST_vae/model_reg_']

names = ['baseline', 'contractive', 'legendre', 'vae']

#cj = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
flh = transforms.RandomHorizontalFlip(p=1.)
flv = transforms.RandomVerticalFlip(p=1.)
#cr = transforms.RandomResizedCrop(size=(32, 32))

labels = ['baseline', 'contractive', 'legendre', 'vae']
path_file = '/home/ramana44/autoencoder_regulrization_conf_tasks_hybrid/FMNIST_samples/'
#path_to_dir = '/home/ramana44/autoencoder_regulrization_conf_tasks_hybrid/output_new/'


global_ind = 0

rand_perturb = []
orig_perturb = []
rec_perturb = []
rec_perturb_hybrid = []
path_to_model = paths[2] + str(alpha)+'_'+str(latent_dim)+'_'+str(hidden_size)+'_'+str(frac)
#model = AE([1,32,32], hidden_size, latent_dim, 3, Sin()).to(device)
#model.load_state_dict(torch.load(path_to_model)['model']) 


#importing my models

from models import AE
from activations import Sin

path_hyb = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_space/output/MRT_full/test_run_saving/'
path_unhyb = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'

#specify hyperparameters
#reg_nodes_sampling = 'legendre'
#alpha = 0.1
#frac = 0.4
#hidden_size = 100
#deg_poly = 20
#latent_dim = 4
#lr = 0.0001
#no_layers = 3
#no_epochs= 100
name_hyb = '_'+reg_nodes_sampling+'_'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)
name_unhyb = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)


inp_dim_hyb = (Hybrid_poly_deg+1)*(Hybrid_poly_deg+1)

inp_dim_unhyb = [1,32,32]

RK_model_reg = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
RK_model_base = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

RK_model_reg.load_state_dict(torch.load(path_hyb+'model_reg'+str(coeff_sol_method)+str(Hybrid_poly_deg)+name_hyb, map_location=torch.device(device)))
RK_model_base.load_state_dict(torch.load(path_hyb+'model_base'+str(coeff_sol_method)+str(Hybrid_poly_deg)+name_hyb, map_location=torch.device(device)))

#importing my models done

'''from convAE import ConvoAE
no_layers_cae = 3
latent_dim_cae = 2
lr_cae =1e-3
name_unhyb_cae = '_'+str(frac)+'_'+str(latent_dim_cae)+'_'+str(lr_cae)+'_'+str(no_layers_cae)
model_convAE = ConvoAE().to(device)
model_convAE.load_state_dict(torch.load(path_unhyb+'model_base_cae_TDA'+name_unhyb_cae, map_location=torch.device(device)), strict=False)'''



model = model_convAE


rec = model(torch.from_numpy(orig).reshape(1,1,32,32).to(device))

print('rec.shape', rec.shape)

#model = model_base
#orig = np.load(path_file+labels[2]+'_'+str(hidden_size)+'_'+str(alpha)+'_'+str(frac)+'_'+str(latent_dim)+'.npy')
#orig = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images.pt')
#orig = np.array(orig[ChoosenImageIndex])
#print(orig.shape)
#orig = orig.reshape(32,32)
#orig_cj = cj(torch.from_numpy(orig).reshape(1,32,32).to('cuda'))

#orig = orig.reshape(1,1024)

orig_flh = flh(torch.from_numpy(orig).reshape(1,32,32).to(device))
orig_flv = flv(torch.from_numpy(orig).reshape(1,32,32).to(device))
#orig_cr = cr(torch.from_numpy(orig).reshape(1,32,32).to('cuda'))
#orig_rot = torchvision.transforms.functional.rotate(torch.from_numpy(orig).reshape(1,32,32).to('cuda'), angle=60)
for proz in prozs:
    rand_perturb.append(np.random.rand(1, 1024)*(np.max(orig)-np.min(orig))*proz)
print((rand_perturb[0].shape))
#rec = model(torch.from_numpy(orig).reshape(1,32,32).to('cuda'))
#rec_cj = model(cj(torch.from_numpy(orig).reshape(1,32,32).to('cuda')))
rec_flv = model(torch.tensor(flv(torch.from_numpy(orig).reshape(1,1,32,32))).to(device))
rec_flh = model(flh(torch.from_numpy(orig).reshape(1,1,32,32).to(device)))
#rec_cr = model(cr(torch.from_numpy(orig).reshape(1,32,32).to('cuda')))
#rec_rot = model(torchvision.transforms.functional.rotate(torch.from_numpy(orig).reshape(1,32,32).to('cuda'), angle=60))
for rand_transform in rand_perturb:
    orig_perturb.append(torch.from_numpy(np.add(orig,rand_transform)).reshape(1,1,32,32).to(device))
    rec_perturb.append(model(orig_perturb[-1].float()))





ax[convAE_im_index,0].imshow(rec.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec.detach().cpu().numpy().reshape(32,32)), np.max(orig.reshape(32,32)))
minv = min(np.min(rec.detach().cpu().numpy().reshape(32,32)), np.min(orig.reshape(32,32)))
rec_n = (rec.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig.reshape(32,32) - minv)/(maxv-minv)*255.

ax[convAE_im_index,1].imshow(rec_flh.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flh.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flh.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

ax[convAE_im_index,2].imshow(rec_flv.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flv.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flv.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.

for k in range(len(rand_perturb)):
    ax[convAE_im_index,3+k].imshow(rec_perturb[k].detach().cpu().numpy().reshape(32,32))
    maxv = max(np.max(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.max(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    minv = min(np.min(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.min(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    rec_n = (rec_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
    orig_n = (orig_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
 
####################################################################################################################


#orig = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images.pt')
#orig = np.array(orig[ChoosenImageIndex])
#orig = orig.reshape(1,1024)
model_reg = RK_model_base

u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
x = np.linspace(-1,1,32)
X_p = u_ob.data_axes([x,x]).T

#now get Runge Kutta coefficients for flipped and noised images before sending them to the autoencoders

#Fr = torch.tensor(orig).reshape(32*32)

def get_all_thetas(listedImage):
    #print('listedImage.shape',listedImage.shape)
    Fr = torch.tensor(listedImage).reshape(32*32)

    '''def grad_x(t,theta):
        theta_t = torch.tensor(theta)
        return -2*torch.matmul(X_p.T,(torch.matmul(X_p,theta_t)-Fr)).detach().numpy()

    def give_theta_t():
        start = time.time()
        u_ob.set_weights_val(0.0)
        theta_0 =  list(u_ob.parameters())[0][0]
        dt = 0.01
        theta_t = theta_0
        for k in range(20):
            theta_int =  scipy.integrate.RK45(grad_x, 0.1, theta_t.detach().numpy(), 100)
            theta_int.step()
            theta_t = torch.tensor(theta_int.y)
        return theta_t

    act_theta = give_theta_t()'''

    get = np.linalg.lstsq(np.array(X_p), listedImage.reshape(32*32), rcond='warn')
    act_theta = torch.tensor(get[0])

    return act_theta

testRK = get_all_thetas(orig)
testRK = testRK.unsqueeze(0)
print('testRK.shape',testRK.shape)

rec_AE = model_reg(testRK.float().to(device)).view(testRK.shape)
#rec_bAE_train = model_base(testRK.float()).view(testRK.shape)

rec_AE = torch.tensor(rec_AE, requires_grad=False)
#rec_bAE_train = torch.tensor(rec_bAE_train, requires_grad=False)

recIM_AE = torch.matmul(X_p.float().to(device), rec_AE.squeeze(1).T.to(device)).T
recIM_AE[np.where(recIM_AE.cpu() < 0.0)] = 0

#recIM_AE = torch.matmul(X_p.float(), rec_AE)

#reconBase_train = torch.matmul(X_p.float(), rec_bAE_train.squeeze(1).T).T

reconReg_train = recIM_AE.reshape(1,32,32)
#reconBase_train = reconBase_train.reshape(1,32,32)


#test getting image back from coeffs and save it

#reconReg_train = torch.matmul(X_p, testRK)
#reconReg_train = reconReg_train.reshape(32,32)
#plt.savefig(reconReg_train, '/home/ramana44/autoencoder_regulrization_conf_tasks_hybrid/output_new/reconReg_train.png')

'''plt.imshow(reconReg_train[0])
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('/home/ramana44/autoencoder_regulrization_conf_tasks_hybrid/output_new/reconReg_train.png')'''
#plt.show()

#print('reconReg_train.shape', reconReg_train.shape)

#print('orig.shape',orig.shape)
orig = orig.reshape(1,1024)

#orig = orig.reshape(32,32)
#orig_cj = cj(torch.from_numpy(orig).reshape(1,32,32).to('cuda'))
orig_flh = flh(torch.from_numpy(orig).reshape(1,32,32).to(device))
orig_flv = flv(torch.from_numpy(orig).reshape(1,32,32).to(device))
#orig_cr = cr(torch.from_numpy(orig).reshape(1,32,32).to('cuda'))
#orig_rot = torchvision.transforms.functional.rotate(torch.from_numpy(orig).reshape(1,32,32).to('cuda'), angle=60)
for proz in prozs:
    rand_perturb.append(np.random.rand(1, 1024)*(np.max(orig)-np.min(orig))*proz)
print((rand_perturb[0].shape))
#rec = model(torch.from_numpy(orig).reshape(1,32,32).to(device))
rec = reconReg_train
#rec = model_reg(torch.from_numpy(orig).reshape(1,32,32).to(device))
#rec_cj = model(cj(torch.from_numpy(orig).reshape(1,32,32).to('cuda')))

vertical_filpped = flv(torch.from_numpy(orig).reshape(1,32,32).to(device))
horizontal_filpped = flh(torch.from_numpy(orig).reshape(1,32,32).to(device))
print('checkFlip.shape', vertical_filpped.shape)

vertical_filpped_coeffs = get_all_thetas(vertical_filpped.cpu())
horizontal_filpped_coeffs = get_all_thetas(horizontal_filpped.cpu())

vertical_filpped_coeffs = vertical_filpped_coeffs.unsqueeze(0)
horizontal_filpped_coeffs = horizontal_filpped_coeffs.unsqueeze(0)

hyb_rec_flv_coeffs = model_reg(vertical_filpped_coeffs.float().to(device)).view(vertical_filpped_coeffs.shape)
hyb_rec_flh_coeffs = model_reg(horizontal_filpped_coeffs.float().to(device)).view(horizontal_filpped_coeffs.shape)

hyb_rec_flv_coeffs = torch.tensor(hyb_rec_flv_coeffs, requires_grad=False)
hyb_rec_flh_coeffs = torch.tensor(hyb_rec_flh_coeffs, requires_grad=False)

recIM_flv = torch.matmul(X_p.float().to(device), hyb_rec_flv_coeffs.squeeze(1).T.to(device)).T
recIM_flv[np.where(recIM_flv.cpu() < 0.0)] = 0


recIM_flh = torch.matmul(X_p.float().to(device), hyb_rec_flh_coeffs.squeeze(1).T.to(device)).T
recIM_flh[np.where(recIM_flh.cpu() < 0.0)] = 0

hyb_rec_flv = recIM_flv.reshape(1,32,32)
hyb_rec_flh = recIM_flh.reshape(1,32,32)



rec_flh = hyb_rec_flh
rec_flv = hyb_rec_flv


print('rec_flv.shape', rec_flv.shape)
orig_perturb_Allcoeffs = []

for rand_transform in rand_perturb:
    orig_perturb.append(torch.from_numpy(np.add(orig,rand_transform)).reshape(1,32,32).to(device))
    #print('orig_perturb[0].shape', orig_perturb[0].shape)    
    #rec_perturb.append(model(orig_perturb[-1].float()))



    orig_perturb_coeffs = get_all_thetas(orig_perturb[-1].float().cpu())
    orig_perturb_coeffs = orig_perturb_coeffs.unsqueeze(0)

    orig_perturb_Allcoeffs.append(orig_perturb_coeffs)
    reconed_coeffs = model_reg(orig_perturb_coeffs.float().to(device))
    reconed_coeffs = torch.tensor(reconed_coeffs, requires_grad=False)
    
    reconed_images = torch.matmul(X_p.float().to(device), reconed_coeffs.squeeze(1).T.to(device)).T
    reconed_images[np.where(reconed_images.cpu() < 0.0)] = 0

    #reconed_images = reconed_images.reshape(1,32,32)
    print(reconed_images.shape)

    rec_perturb_hybrid.append(reconed_images)
    #print('rec_perturb[0].shape', rec_perturb[0].shape)    
    print('orig_perturb_coeffs.shape',orig_perturb_coeffs.shape)

#rec_perturb = rec_perturb_hybrid



'''ax[3,0].imshow(rec.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec.detach().cpu().numpy().reshape(32,32)), np.max(orig.reshape(32,32)))
minv = min(np.min(rec.detach().cpu().numpy().reshape(32,32)), np.min(orig.reshape(32,32)))
rec_n = (rec.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig.reshape(32,32) - minv)/(maxv-minv)*255.


ax[3,1].imshow(rec_flh.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flh.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flh.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flh.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flh.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.


ax[3,2].imshow(rec_flv.detach().cpu().numpy().reshape(32,32))
maxv = max(np.max(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.max(orig_flv.detach().cpu().numpy().reshape(32,32)))
minv = min(np.min(rec_flv.detach().cpu().numpy().reshape(32,32)), 
            np.min(orig_flv.detach().cpu().numpy().reshape(32,32)))
rec_n = (rec_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
orig_n = (orig_flv.detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.


for k in range(len(rand_perturb)):
    ax[3,3+k].imshow(rec_perturb[k].detach().cpu().numpy().reshape(32,32))
    maxv = max(np.max(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.max(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    minv = min(np.min(rec_perturb[k].detach().cpu().numpy().reshape(32,32)), 
                np.min(orig_perturb[k].detach().cpu().numpy().reshape(32,32)))
    rec_n = (rec_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.
    orig_n = (orig_perturb[k].detach().cpu().numpy().reshape(32,32) - minv)/(maxv-minv)*255.'''


###############################################################################################################
rand_perturb = []
orig_perturb = []
rec_perturb = []
model_reg = RK_model_reg
#orig = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images.pt')
#orig = np.array(orig[ChoosenImageIndex])

u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
x = np.linspace(-1,1,32)
X_p = u_ob.data_axes([x,x]).T

#now get Runge Kutta coefficients for flipped and noised images before sending them to the autoencoders

#Fr = torch.tensor(orig).reshape(32*32)

'''def get_all_thetas(listedImage):
    #print('listedImage.shape',listedImage.shape)
    Fr = torch.tensor(listedImage).reshape(32*32)

    def grad_x(t,theta):
        theta_t = torch.tensor(theta)
        return -2*torch.matmul(X_p.T,(torch.matmul(X_p,theta_t)-Fr)).detach().numpy()

    def give_theta_t():
        start = time.time()
        u_ob.set_weights_val(0.0)
        theta_0 =  list(u_ob.parameters())[0][0]
        dt = 0.01
        theta_t = theta_0
        for k in range(20):
            theta_int =  scipy.integrate.RK45(grad_x, 0.1, theta_t.detach().numpy(), 100)
            theta_int.step()
            theta_t = torch.tensor(theta_int.y)
        return theta_t

    act_theta = give_theta_t()
    return act_theta'''

testRK = get_all_thetas(orig)
testRK = testRK.unsqueeze(0)
print('testRK.shape',testRK.shape)

rec_AE = model_reg(testRK.float().to(device)).view(testRK.shape)
#rec_bAE_train = model_base(testRK.float()).view(testRK.shape)

rec_AE = torch.tensor(rec_AE, requires_grad=False)
#rec_bAE_train = torch.tensor(rec_bAE_train, requires_grad=False)

recIM_AE = torch.matmul(X_p.float().to(device), rec_AE.squeeze(1).T.to(device)).T
recIM_AE[np.where(recIM_AE.cpu() < 0.0)] = 0


#recIM_AE = torch.matmul(X_p.float(), rec_AE)

#reconBase_train = torch.matmul(X_p.float(), rec_bAE_train.squeeze(1).T).T

reconReg_train = recIM_AE.reshape(1,32,32)
#reconBase_train = reconBase_train.reshape(1,32,32)


#test getting image back from coeffs and save it

#reconReg_train = torch.matmul(X_p, testRK)
#reconReg_train = reconReg_train.reshape(32,32)
#plt.savefig(reconReg_train, '/home/ramana44/autoencoder_regulrization_conf_tasks_hybrid/output_new/reconReg_train.png')

'''plt.imshow(reconReg_train[0])
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('/home/ramana44/autoencoder_regulrization_conf_tasks_hybrid/output_new/reconReg_train.png')'''
#plt.show()

#print('reconReg_train.shape', reconReg_train.shape)

#print('orig.shape',orig.shape)
orig = orig.reshape(1,1024)

#orig = orig.reshape(32,32)
#orig_cj = cj(torch.from_numpy(orig).reshape(1,32,32).to('cuda'))
orig_flh = flh(torch.from_numpy(orig).reshape(1,32,32).to(device))
orig_flv = flv(torch.from_numpy(orig).reshape(1,32,32).to(device))
#orig_cr = cr(torch.from_numpy(orig).reshape(1,32,32).to('cuda'))
#orig_rot = torchvision.transforms.functional.rotate(torch.from_numpy(orig).reshape(1,32,32).to('cuda'), angle=60)
for proz in prozs:
    rand_perturb.append(np.random.rand(1, 1024)*(np.max(orig)-np.min(orig))*proz)
print((rand_perturb[0].shape))
#rec = model(torch.from_numpy(orig).reshape(1,32,32).to(device))
rec = reconReg_train
#rec = model_reg(torch.from_numpy(orig).reshape(1,32,32).to(device))
#rec_cj = model(cj(torch.from_numpy(orig).reshape(1,32,32).to('cuda')))

vertical_filpped = flv(torch.from_numpy(orig).reshape(1,32,32).to(device))
horizontal_filpped = flh(torch.from_numpy(orig).reshape(1,32,32).to(device))
print('checkFlip.shape', vertical_filpped.shape)

vertical_filpped_coeffs = get_all_thetas(vertical_filpped.cpu())
horizontal_filpped_coeffs = get_all_thetas(horizontal_filpped.cpu())

vertical_filpped_coeffs = vertical_filpped_coeffs.unsqueeze(0)
horizontal_filpped_coeffs = horizontal_filpped_coeffs.unsqueeze(0)

hyb_rec_flv_coeffs = model_reg(vertical_filpped_coeffs.float().to(device)).view(vertical_filpped_coeffs.shape)
hyb_rec_flh_coeffs = model_reg(horizontal_filpped_coeffs.float().to(device)).view(horizontal_filpped_coeffs.shape)

hyb_rec_flv_coeffs = torch.tensor(hyb_rec_flv_coeffs, requires_grad=False)
hyb_rec_flh_coeffs = torch.tensor(hyb_rec_flh_coeffs, requires_grad=False)

recIM_flv = torch.matmul(X_p.float().to(device), hyb_rec_flv_coeffs.squeeze(1).T.to(device)).T

recIM_flv[np.where(recIM_flv.cpu() < 0.0)] = 0


recIM_flh = torch.matmul(X_p.float().to(device), hyb_rec_flh_coeffs.squeeze(1).T.to(device)).T

recIM_flh[np.where(recIM_flh.cpu() < 0.0)] = 0


hyb_rec_flv = recIM_flv.reshape(1,32,32)
hyb_rec_flh = recIM_flh.reshape(1,32,32)



rec_flh = hyb_rec_flh
rec_flv = hyb_rec_flv


print('rec_flv.shape', rec_flv.shape)
orig_perturb_Allcoeffs = []
rec_perturb_hybrid = []
for rand_transform in rand_perturb:
    orig_perturb.append(torch.from_numpy(np.add(orig,rand_transform)).reshape(1,32,32).to(device))
    #print('orig_perturb[0].shape', orig_perturb[0].shape)    
    #rec_perturb.append(model(orig_perturb[-1].float()))



    orig_perturb_coeffs = get_all_thetas(orig_perturb[-1].float().cpu())
    orig_perturb_coeffs = orig_perturb_coeffs.unsqueeze(0)

    orig_perturb_Allcoeffs.append(orig_perturb_coeffs)
    reconed_coeffs = model_reg(orig_perturb_coeffs.float().to(device))
    reconed_coeffs = torch.tensor(reconed_coeffs, requires_grad=False)
    
    reconed_images = torch.matmul(X_p.float().to(device), reconed_coeffs.squeeze(1).T.to(device)).T

    reconed_images[np.where(reconed_images.cpu() < 0.0)] = 0

    #reconed_images = reconed_images.reshape(1,32,32)
    print(reconed_images.shape)

    rec_perturb_hybrid.append(reconed_images)
    #print('rec_perturb[0].shape', rec_perturb[0].shape)    
    print('orig_perturb_coeffs.shape',orig_perturb_coeffs.shape)

rec_perturb = rec_perturb_hybrid

#hyb_AEREG_inm_ind = 4

ax[hyb_AEREG_inm_ind,0].imshow(rec.detach().cpu().numpy().reshape(32,32))



ax[hyb_AEREG_inm_ind,1].imshow(rec_flh.detach().cpu().numpy().reshape(32,32))



ax[hyb_AEREG_inm_ind,2].imshow(rec_flv.detach().cpu().numpy().reshape(32,32))



for k in range(len(rand_perturb)):
    ax[hyb_AEREG_inm_ind,3+k].imshow(rec_perturb[k].detach().cpu().numpy().reshape(32,32))



#################################################################################################################

for i in range(8):
    for j in range(3+len(orig_perturb)):
        ax[i,j].set_axis_off()


'''plt.subplots_adjust(left=0.01,
                    bottom=0.01, 
                    right=0.99, 
                    top=0.99, 
                    wspace=0.03, 
                    hspace=0.03)'''

plt.subplots_adjust(left=0.01,
                    bottom=0.01, 
                    right=0.40, 
                    top=1.0, 
                    wspace=0.0005, 
                    hspace=0.05)




plt.savefig(path_to_dir+'Updated_RK_trans_FMNIST'+str(coeff_sol_method)+'CoffPolDeg_'+str(Hybrid_poly_deg)+'_'+choosenData+'_'+'ImagNo_'+str(ChoosenImageIndex)+'_'+str(hidden_size)+
                        '_'+str(alpha)+'_'+str(frac)+'_LatDim'+str(latent_dim)+'.png',
            dpi=fig.dpi, bbox_inches='tight', pad_inches=0.10, edgecolor='w', facecolor="w")
plt.close()