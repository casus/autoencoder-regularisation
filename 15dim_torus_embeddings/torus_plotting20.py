import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from models_of_VAEs_circle import BetaVAE, MLPVAE, ConvVAE_circle, VAE_mlp_circle_new
#from mlp_vae_model import VariationalAutoencoder

import numpy as np
import matplotlib.pyplot as plt

import os
import re
#from datasets import getMNIST, getFashionMNIST, getCifar10, getDataset
import copy

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cpu')
device = torch.device('cuda')
from models import AE
#from vae import BetaVAE
from activations import Sin
#from regularisers import computeC1Loss
#from models_circle import MLPVAE
import matplotlib
matplotlib.rcdefaults()
#from tabulate import tabulate

#from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import peak_signal_noise_ratio as psnr

filename = "table_model_reg_legendre_legendre.txt"
latent_dim = 3




class Autoencoder_linear(nn.Module):
    def __init__(self,latent_dim):

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(15, 6),  #input layer
            Sin(),
            nn.Linear(6, 6),    #h1
            Sin(),
            nn.Linear(6, 6),    #h1
            Sin(),
            nn.Linear(6,latent_dim),  # latent layer
            Sin()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 6),  #input layer
            Sin(),
            nn.Linear(6, 6),    #h1
            Sin(),
            nn.Linear(6, 6),    #h1
            Sin(),
            nn.Linear(6, 15),  # latent layer
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

ChoosenImageIndex = 3

#Hybrid AE-REG parameters
Hybrid_poly_deg = 16


# MLPAE and AE-REG parameters
hidden_size = 6
alpha = 0.5
frac = 1.0

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
#path_hyb = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_space/output/MRT_full/test_run_saving/'
path_unhyb = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'

#specify hyperparameters
reg_nodes_sampling = 'legendre'

deg_poly = 21
deg_poly_later = 45
lr = 0.002
no_layers = 2
no_epochs= 100

name_unhyb = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

name_unhyb_later = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly_later)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

#name_unhyb = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)


#inp_dim_hyb = (Hybrid_poly_deg+1)*(Hybrid_poly_deg+1)

inp_dim_unhyb = 15

model_reg = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
model_base = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

model_reg.load_state_dict(torch.load(path_unhyb+'model_reg_tor20'+name_unhyb, map_location=torch.device(device)))
model_base.load_state_dict(torch.load(path_unhyb+'model_base_tor20'+name_unhyb, map_location=torch.device(device)))


print('check Loaded? ')

# 3 training points
training_points = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/torus_dataset/15_dim_test_data_torus_1000.pt').to(device)
training_points = training_points.cpu()
# Generate 15 dim circle and normalize it

num_pts = 500
radius = 1.0
angles = np.linspace(0.,360.,num_pts)
angles_in_rad = (angles*22)/(7*180)
x_arr = radius* np.cos(angles_in_rad)
y_arr = radius* np.sin(angles_in_rad)
points = np.dstack((x_arr, y_arr))
circle_points = points.reshape(num_pts,2)

sphere_coords_train = torch.tensor(circle_points)
A_transform5 = np.random.uniform(-2, 2, 2*15).reshape(2, 15)
A_transform5 = torch.tensor(A_transform5)
fifteen_dim_sphere_train = torch.matmul(sphere_coords_train, A_transform5.double())



#now normalizing
fifteen_dim_sphere_train =-1 + 2*( (fifteen_dim_sphere_train - fifteen_dim_sphere_train.min())/(fifteen_dim_sphere_train.max() - fifteen_dim_sphere_train.min()) )


#perturbing after normalizing

proz = 0.0
fifteen_dim_sphere_train = np.array(fifteen_dim_sphere_train)
print('fifteen_dim_sphere_train.shape', fifteen_dim_sphere_train.shape)
rand_transform = np.random.rand(num_pts,15)*(np.max(fifteen_dim_sphere_train)-np.min(fifteen_dim_sphere_train))*proz

fifteen_dim_sphere_train = torch.from_numpy(np.add(fifteen_dim_sphere_train,rand_transform)).reshape(num_pts,15)

fifteen_dim_sphere_train = torch.tensor(fifteen_dim_sphere_train)
fifteen_dim_sphere_train = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/torus_dataset/15_dim_test_data_torus_500.pt')

batch_x = fifteen_dim_sphere_train.to(device)
rec_rAE = model_reg.encoder(batch_x.float()).detach().cpu().numpy() 
rec_bAE = model_base.encoder(batch_x.float()).detach().cpu().numpy() 


print('training_points.shape', training_points.shape)

print('rec_rAE.shape', rec_rAE.shape)

print('rec_bAE.shape', rec_bAE.shape)

torch.save(rec_rAE, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_torus_embeddings/embedded_point_cloud_saved/20trainingPoints/rec_rAE.pt')

torch.save(rec_bAE, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_torus_embeddings/embedded_point_cloud_saved/20trainingPoints/rec_bAE.pt')

'''plt.scatter(sphere_coords_train[:,0], sphere_coords_train[:,1], color='gray')
plt.scatter(rec_rAE[:,0], rec_rAE[:,1], color = 'orange')
plt.scatter(training_points[:,0], training_points[:,1], color='blue')
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_circle_embedding_plots/tr1_AE_REG.png')
plt.close()


plt.scatter(sphere_coords_train[:,0], sphere_coords_train[:,1], color='gray')
plt.scatter(rec_bAE[:,0], rec_bAE[:,1], color = 'orange')
plt.scatter(training_points[:,0], training_points[:,1], color='blue')
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_circle_embedding_plots/tr1_MLPAE.png')
plt.close()'''











#loading convolutional autoencoder
from circle_all_models import ConvoAE
no_layers_cae = 2
latent_dim_cae = latent_dim
lr_cae =0.002
name_unhyb_cae = '_'+str(frac)+'_'+str(latent_dim_cae)+'_'+str(lr_cae)+'_'+str(no_layers_cae)
model_convAE = ConvoAE(latent_dim_cae).to(device)
model_convAE.load_state_dict(torch.load(path_unhyb+'model_base_cae_tor20'+name_unhyb_cae, map_location=torch.device(device)), strict=False)


print("loadin cricle cnn ?")


model_convAE = model_convAE.encoder(batch_x.reshape(num_pts,1,15).float()).detach().cpu().numpy() 
#rec_bAE = model_base.encoder(batch_x.float()).detach().cpu().numpy() 
print('model_convAE.shape', model_convAE.shape)
torch.save(model_convAE, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_torus_embeddings/embedded_point_cloud_saved/20trainingPoints/model_convAE.pt')


'''plt.scatter(sphere_coords_train[:,0], sphere_coords_train[:,1], color='gray')
plt.scatter(model_convAE[:,0], model_convAE[:,1], color = 'orange')
plt.scatter(training_points[:,0], training_points[:,1], color='blue')
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_circle_embedding_plots/tr1_CNN.png')
plt.close()'''



#loading contractive autoencoder
no_layers_contraae = 2
latent_dim_contraae = latent_dim
lr_contraae =0.002
name_unhyb_contraae = '_'+str(frac)+'_'+str(latent_dim_contraae)+'_'+str(lr_contraae)+'_'+str(no_layers_contraae)
model_contra = Autoencoder_linear(latent_dim_contraae).to(device)
model_contra.load_state_dict(torch.load(path_unhyb+'model_base_contraAE_tor20'+name_unhyb_contraae, map_location=torch.device(device)), strict=False)

print("loading ?")

model_contra = model_contra.encoder(batch_x.reshape(num_pts,1,15).float()).detach().cpu().numpy() 
#rec_bAE = model_base.encoder(batch_x.float()).detach().cpu().numpy() 
model_contra = model_contra.reshape(num_pts,3)
print("contra shape ", model_contra.shape)

torch.save(model_contra, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_torus_embeddings/embedded_point_cloud_saved/20trainingPoints/model_contra.pt')


'''plt.scatter(sphere_coords_train[:,0], sphere_coords_train[:,1], color='gray')
plt.scatter(model_contra[:,0], model_contra[:,1], color = 'orange')
plt.scatter(training_points[:,0], training_points[:,1], color='blue')
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_circle_embedding_plots/tr1_contra.png')
plt.close()'''



#from vae import BetaVAE
from activations import Sin
activation = Sin()
lr = 1e-4
name_unhyb_vae = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

model_mlpVAE = MLPVAE(15, hidden_size, latent_dim, 
                    no_layers, activation).to(device) # regularised autoencoder

print("loaded mlpvae circle ?")


INPUT_DIM = 15
H_DIM = 6
Z_DIM = 2
#lr = 0.0001

#model_mlpVAE = VariationalAutoencoder(INPUT_DIM, H_DIM, Z_DIM).to(device)
image_size = 15
h_dim = 6

model_mlpVAE = VAE_mlp_circle_new(image_size, h_dim, z_dim=latent_dim).to(device)

#model_betaVAE = BetaVAE_(15, no_filters=5, no_layers=2,
              #kernel_size=3, latent_dim=2, activation = Sin()).to(device) # regularised autoencoder


'''model_mlpVAE = MLPVAE(15, hidden_size, latent_dim, 
                    no_layers, activation).to(device) # regularised autoencoder

model_betaVAE = BetaVAE(15, no_filters=4, no_layers=3,
                kernel_size=3, latent_dim=10, activation = Sin()).to(device) # regularised autoencoder'''

#model_betaVAE = BetaVAE(batch_size = 1, img_depth = 1, net_depth = no_layers, z_dim = latent_dim, img_dim = 32).to(device)
model_mlpVAE.load_state_dict(torch.load(path_unhyb+'model_base_mlp_vae_tor20'+name_unhyb_vae, map_location=torch.device(device)), strict=False)

model_betaVAE = ConvVAE_circle(image_channels=1, h_dim=5*4, z_dim=latent_dim).to(device)
model_betaVAE.load_state_dict(torch.load(path_unhyb+'model_base_cnn_vae_tor20'+name_unhyb_vae, map_location=torch.device(device)), strict=False)

#model_mlpVAE = model_mlpVAE.encode(batch_x.reshape(num_pts,1,15).float()).detach().cpu().numpy() 

'''model_mlpVAE = model_mlpVAE.encoder(batch_x.reshape(num_pts,1,15).float())


print('shape here', len(model_mlpVAE[0]))

model_mlpVAE = torch.tensor(model_mlpVAE[0]).reshape(num_pts,2).cpu()

print('shape here again', model_mlpVAE.shape)

plt.scatter(model_mlpVAE[:,0], model_mlpVAE[:,1])
#plt.scatter(circle_points[:,0], circle_points[:,1])
plt.savefig('/home/ramana44/FashionMNIST5LayersTrials/circle_plots/tr1_mlpvae.png')
plt.close()'''


model_mlpVAE = model_mlpVAE.fc1(model_mlpVAE.encoder(batch_x.reshape(num_pts,1,15).float())).reshape(num_pts,3).detach().cpu().numpy() 

print('model_mlpVAE.shape', model_mlpVAE.shape)

torch.save(model_mlpVAE, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_torus_embeddings/embedded_point_cloud_saved/20trainingPoints/model_mlpVAE.pt')


'''plt.scatter(sphere_coords_train[:,0], sphere_coords_train[:,1], color='gray')
plt.scatter(model_mlpVAE[:,0], model_mlpVAE[:,1], color = 'orange')
plt.scatter(training_points[:,0], training_points[:,1], color='blue')
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_circle_embedding_plots/tr1_mlpvae.png')
plt.close()'''


model_betaVAE = model_betaVAE.fc1(model_betaVAE.encoder(batch_x.reshape(num_pts,1,15).float())).detach().cpu().numpy() 
sphere_coords_train


'''plt.scatter(sphere_coords_train[:,0], sphere_coords_train[:,1], color='gray')
plt.scatter(model_betaVAE[:,0], model_betaVAE[:,1], color = 'orange')
plt.scatter(training_points[:,0], training_points[:,1], color='blue')
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_circle_embedding_plots/tr1_cnnvae.png')
plt.close()'''

print("loadin cricle vaes ?")

print('model_betaVAE.shape', model_betaVAE.shape)

torch.save(model_betaVAE, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/15dim_torus_embeddings/embedded_point_cloud_saved/20trainingPoints/model_betaVAE.pt')
