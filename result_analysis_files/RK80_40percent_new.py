import sys
sys.path.append('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders')


from get_data import get_data, get_data_train, get_data_val
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


deg_quad = 80
u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
x = np.linspace(-1,1,96)
X_p = u_ob.data_axes([x,x]).T


# Get RK coefficients for these images and also the images

#before executing this cell you need to have preexisting trained coefficients for what ever degree you want

trainImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/trainDataSet.pt',map_location=torch.device('cpu'))
trainCoeffs = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/trainDataRK_coeffs.pt',map_location=torch.device('cpu'))


testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/testDataSet.pt',map_location=torch.device('cpu'))
testCoeffs = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/testDataRK_coeffs.pt',map_location=torch.device('cpu'))

Analys_size = 50

trainImages = trainImages[:Analys_size]
trainCoeffs = trainCoeffs[:Analys_size]

testImages = testImages[:Analys_size]
testCoeffs = testCoeffs[:Analys_size]

# load trained rAE and bAE

#from models_un import AE_un
from models import AE
from activations import Sin

#path_hyb = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_space/output/MRT_full/test_run_saving/'
#path_unhyb = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'

path_unhyb = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/NonHybridOasisMRI_AE-REG/output/MRT_full/test_run_saving/'
path_hyb = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/output/MRT_full/test_run_saving/'


#specify hyperparameters
reg_nodes_sampling = 'legendre'
alpha = 0.1
frac = 0.8
hidden_size = 1000
deg_poly = 20
deg_poly_forRK = 20
latent_dim = 40
lr = 0.0001
no_layers = 5
#no_epochs= 100
#name_hyb = '_'+reg_nodes_sampling+'_'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly_forRK)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

#name_unhyb = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)
name_hyb = '_'+reg_nodes_sampling+'_'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)
name_unhyb = '_'+reg_nodes_sampling+'_'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

#no_channels, dx, dy = (train_loader_alz.dataset.__getitem__(1).shape)
#inp_dim = [no_channels, dx-21, dy-21]
inp_dim_hyb = (deg_quad+1)*(deg_quad+1)

inp_dim_unhyb = [1,96,96]

RK_model_reg = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
RK_model_base = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

model_reg = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
model_base = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

#model_reg.load_state_dict(torch.load(path+'model_reg'+name, map_location=torch.device('cpu'))["model"])
#model_base.load_state_dict(torch.load(path+'model_reg'+name, map_location=torch.device('cpu'))["model"])

RK_model_reg.load_state_dict(torch.load(path_hyb+'model_reg'+name_hyb, map_location=torch.device('cpu')))
RK_model_base.load_state_dict(torch.load(path_hyb+'model_base'+name_hyb, map_location=torch.device('cpu')))

model_reg.load_state_dict(torch.load(path_unhyb+'model_reg'+name_unhyb, map_location=torch.device('cpu')))
model_base.load_state_dict(torch.load(path_unhyb+'model_base'+name_unhyb, map_location=torch.device('cpu')))
#model_reg.eval()
#model_base.eval()

unhyb_rec_rAE_train = model_reg(trainImages).view(trainImages.shape).detach().numpy() 
unhyb_rec_bAE_train = model_base(trainImages).view(trainImages.shape).detach().numpy() 

unhyb_rec_rAE_test = model_reg(testImages).view(testImages.shape).detach().numpy() 
unhyb_rec_bAE_test = model_base(testImages).view(testImages.shape).detach().numpy() 


unhyb_rec_rAE_train = torch.tensor(unhyb_rec_rAE_train, requires_grad=False)
unhyb_rec_bAE_train = torch.tensor(unhyb_rec_bAE_train, requires_grad=False)

unhyb_rec_rAE_test = torch.tensor(unhyb_rec_rAE_test, requires_grad=False)
unhyb_rec_bAE_test = torch.tensor(unhyb_rec_bAE_test, requires_grad=False)

rec_rAE_train = RK_model_reg(trainCoeffs.float()).view(trainCoeffs.shape)
rec_bAE_train = RK_model_base(trainCoeffs.float()).view(trainCoeffs.shape)

rec_rAE_test = RK_model_reg(testCoeffs.float()).view(testCoeffs.shape)
rec_bAE_test = RK_model_base(testCoeffs.float()).view(testCoeffs.shape)

# reconstruction loss after training the model completely
loss_tre = torch.mean(((unhyb_rec_rAE_test - testImages)**2)*0.5)
# reconstruction loss after training the model completely
loss_tre = torch.mean(((unhyb_rec_bAE_test - testImages)**2)*0.5)

rec_rAE_train = torch.tensor(rec_rAE_train, requires_grad=False)
rec_bAE_train = torch.tensor(rec_bAE_train, requires_grad=False)

rec_rAE_test = torch.tensor(rec_rAE_test, requires_grad=False)
rec_bAE_test = torch.tensor(rec_bAE_test, requires_grad=False)

reconReg_train = torch.matmul(X_p.float(), rec_rAE_train.squeeze(1).T).T

reconBase_train = torch.matmul(X_p.float(), rec_bAE_train.squeeze(1).T).T

reconReg_test = torch.matmul(X_p.float(), rec_rAE_test.squeeze(1).T).T

reconBase_test = torch.matmul(X_p.float(), rec_bAE_test.squeeze(1).T).T

hybrd_reconReg_train = reconReg_train.reshape(Analys_size,1,96,96)
hybrd_reconBase_train = reconBase_train.reshape(Analys_size,1,96,96)

hybrd_reconReg_test = reconReg_test.reshape(Analys_size,1,96,96)
hybrd_reconBase_test = reconBase_test.reshape(Analys_size,1,96,96)

# What if I get rid of negatives in unhybridized ?
'''rec_rAE_train[np.where(rec_rAE_train < 0.0)] = 0
rec_bAE_train[np.where(rec_bAE_train < 0.0)] = 0
rec_rAE_test[np.where(rec_rAE_test < 0.0)] = 0
rec_bAE_test[np.where(rec_bAE_test < 0.0)] = 0'''
# Doesn't make sense to remove negatives from unhybrid because they are coming from sine activation function. Normalization is sufficient
# Getting rid of negatives
hybrd_reconReg_train[np.where(hybrd_reconReg_train < 0.0)] = 0
hybrd_reconBase_train[np.where(hybrd_reconBase_train < 0.0)] = 0
hybrd_reconReg_test[np.where(hybrd_reconReg_test < 0.0)] = 0
hybrd_reconBase_test[np.where(hybrd_reconBase_test < 0.0)] = 0

#unhyb_base_prturb4_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb4_recons_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'.pt')
test = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb4_recons_baseline_Lat40_TDA0.4lot.pt')

print('test.shape', test.shape)

unhyb_base_prturb4_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb4_recons_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
unhyb_base_prturb4_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb4_recons_baseline_Lat40_TDA0.8lot.pt')
#/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb4_recons_baseline_Lat         40        _TDA   0.4       lot.pt
unhyb_base_prturb4_recon = unhyb_base_prturb4_recon[:50]
print('unhyb_base_prturb4_recon.shape',unhyb_base_prturb4_recon.shape)
unhyb_base_prturb4_recon = unhyb_base_prturb4_recon.reshape(50, 1, 96, 96)
unhyb_base_prturb4_recon = torch.tensor(unhyb_base_prturb4_recon, requires_grad=False)
unhyb_reg_prturb4_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb4_recons_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt') 
unhyb_reg_prturb4_recon = unhyb_reg_prturb4_recon[:50]

unhyb_reg_prturb4_recon = unhyb_reg_prturb4_recon.reshape(50, 1, 96, 96)
unhyb_reg_prturb4_recon = torch.tensor(unhyb_reg_prturb4_recon, requires_grad=False)



hyb_base_prturb4_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_RK'+str(deg_quad)+'_all_perturb4_recons_RK_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
hyb_base_prturb4_recon = hyb_base_prturb4_recon.reshape(50, 1, 96, 96)
hyb_base_prturb4_recon = torch.tensor(hyb_base_prturb4_recon, requires_grad=False)
hyb_reg_prturb4_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_RK'+str(deg_quad)+'_all_perturb4_recons_RK_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
hyb_reg_prturb4_recon = hyb_reg_prturb4_recon.reshape(50, 1, 96, 96)
hyb_reg_prturb4_recon = torch.tensor(hyb_reg_prturb4_recon, requires_grad=False)

print("worked?")


unhyb_base_prturb3_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb3_recons_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
unhyb_base_prturb3_recon = unhyb_base_prturb3_recon[:50]
unhyb_base_prturb3_recon = unhyb_base_prturb3_recon.reshape(50, 1, 96, 96)
unhyb_base_prturb3_recon = torch.tensor(unhyb_base_prturb3_recon, requires_grad=False)
unhyb_reg_prturb3_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb3_recons_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
unhyb_reg_prturb3_recon = unhyb_reg_prturb3_recon[:50]
unhyb_reg_prturb3_recon = unhyb_reg_prturb3_recon.reshape(50, 1, 96, 96)
unhyb_reg_prturb3_recon = torch.tensor(unhyb_reg_prturb3_recon, requires_grad=False)



hyb_base_prturb3_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_RK'+str(deg_quad)+'_all_perturb3_recons_RK_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
hyb_base_prturb3_recon = hyb_base_prturb3_recon.reshape(50, 1, 96, 96)
hyb_base_prturb3_recon = torch.tensor(hyb_base_prturb3_recon, requires_grad=False)
hyb_reg_prturb3_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_RK'+str(deg_quad)+'_all_perturb3_recons_RK_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
hyb_reg_prturb3_recon = hyb_reg_prturb3_recon.reshape(50, 1, 96, 96)
hyb_reg_prturb3_recon = torch.tensor(hyb_reg_prturb3_recon, requires_grad=False)


unhyb_base_prturb2_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb2_recons_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
unhyb_base_prturb2_recon = unhyb_base_prturb2_recon[:50]
unhyb_base_prturb2_recon = unhyb_base_prturb2_recon.reshape(50, 1, 96, 96)
unhyb_base_prturb2_recon = torch.tensor(unhyb_base_prturb2_recon, requires_grad=False)
unhyb_reg_prturb2_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb2_recons_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
unhyb_reg_prturb2_recon = unhyb_reg_prturb2_recon[:50]
unhyb_reg_prturb2_recon = unhyb_reg_prturb2_recon.reshape(50, 1, 96, 96)
unhyb_reg_prturb2_recon = torch.tensor(unhyb_reg_prturb2_recon, requires_grad=False)


hyb_base_prturb2_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_RK'+str(deg_quad)+'_all_perturb2_recons_RK_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
hyb_base_prturb2_recon = hyb_base_prturb2_recon.reshape(50, 1, 96, 96)
hyb_base_prturb2_recon = torch.tensor(hyb_base_prturb2_recon, requires_grad=False)
hyb_reg_prturb2_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_RK'+str(deg_quad)+'_all_perturb2_recons_RK_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
hyb_reg_prturb2_recon = hyb_reg_prturb2_recon.reshape(50, 1, 96, 96)
hyb_reg_prturb2_recon = torch.tensor(hyb_reg_prturb2_recon, requires_grad=False)


unhyb_base_prturb1_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb1_recons_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
unhyb_base_prturb1_recon = unhyb_base_prturb1_recon[:50]
unhyb_base_prturb1_recon = unhyb_base_prturb1_recon.reshape(50, 1, 96, 96)
unhyb_base_prturb1_recon = torch.tensor(unhyb_base_prturb1_recon, requires_grad=False)
unhyb_reg_prturb1_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_all_perturb1_recons_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
unhyb_reg_prturb1_recon = unhyb_reg_prturb1_recon[:50]
unhyb_reg_prturb1_recon = unhyb_reg_prturb1_recon.reshape(50, 1, 96, 96)
unhyb_reg_prturb1_recon = torch.tensor(unhyb_reg_prturb1_recon, requires_grad=False)


hyb_base_prturb1_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_RK'+str(deg_quad)+'_all_perturb1_recons_RK_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
hyb_base_prturb1_recon = hyb_base_prturb1_recon.reshape(50, 1, 96, 96)
hyb_base_prturb1_recon = torch.tensor(hyb_base_prturb1_recon, requires_grad=False)
hyb_reg_prturb1_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/MRI_RK'+str(deg_quad)+'_all_perturb1_recons_RK_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
hyb_reg_prturb1_recon = hyb_reg_prturb1_recon.reshape(50, 1, 96, 96)
hyb_reg_prturb1_recon = torch.tensor(hyb_reg_prturb1_recon, requires_grad=False)


# Get rid of negatives again to the perturbed
 



hyb_base_prturb4_recon[np.where(hyb_base_prturb4_recon < 0.0)] = 0
hyb_base_prturb3_recon[np.where(hyb_base_prturb3_recon < 0.0)] = 0
hyb_base_prturb2_recon[np.where(hyb_base_prturb2_recon < 0.0)] = 0
hyb_base_prturb1_recon[np.where(hyb_base_prturb1_recon < 0.0)] = 0


hyb_reg_prturb4_recon[np.where(hyb_reg_prturb4_recon < 0.0)] = 0
hyb_reg_prturb3_recon[np.where(hyb_reg_prturb3_recon < 0.0)] = 0
hyb_reg_prturb2_recon[np.where(hyb_reg_prturb2_recon < 0.0)] = 0
hyb_reg_prturb1_recon[np.where(hyb_reg_prturb1_recon < 0.0)] = 0



from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from matplotlib.colors import Normalize


ssimlists_unhyb_base = []
ssimlists_unhyb_reg = []
ssimlists_hyb_base = []
ssimlists_hyb_reg = []

ssimlists_perturb4_hyb_base = []
ssimlists_perturb4_hyb_reg = []
ssimlists_perturb4_base = []
ssimlists_perturb4_reg = []

ssimlists_perturb3_hyb_base = []
ssimlists_perturb3_hyb_reg = []
ssimlists_perturb3_base = []
ssimlists_perturb3_reg = []

ssimlists_perturb2_hyb_base = []
ssimlists_perturb2_hyb_reg = []
ssimlists_perturb2_base = []
ssimlists_perturb2_reg = []

ssimlists_perturb1_hyb_base = []
ssimlists_perturb1_hyb_reg = []
ssimlists_perturb1_base = []
ssimlists_perturb1_reg = []

##############################################################################################

psnrlists_unhyb_base = []
psnrlists_unhyb_reg = []
psnrlists_hyb_base = []
psnrlists_hyb_reg = []

psnrlists_perturb4_base = []
psnrlists_perturb4_reg = []
psnrlists_perturb4_hyb_base = []
psnrlists_perturb4_hyb_reg = []

psnrlists_perturb3_base = []
psnrlists_perturb3_reg = []
psnrlists_perturb3_hyb_base = []
psnrlists_perturb3_hyb_reg = []

psnrlists_perturb2_base = []
psnrlists_perturb2_reg = []
psnrlists_perturb2_hyb_base = []
psnrlists_perturb2_hyb_reg = []

psnrlists_perturb1_base = []
psnrlists_perturb1_reg = []
psnrlists_perturb1_hyb_base = []
psnrlists_perturb1_hyb_reg = []

for i in range(len(testImages)):

    testImage_normal = Normalize()(testImages[i])
    recon_normal_unhyb_base = Normalize()(unhyb_rec_bAE_test[i])
    recon_normal_unhyb_reg = Normalize()(unhyb_rec_rAE_test[i])
    recon_normal_hyb_base = Normalize()(hybrd_reconBase_test[i])
    recon_normal_hyb_reg = Normalize()(hybrd_reconReg_test[i])

    recon_normal_unhyb_base_perturb4 = Normalize()(unhyb_base_prturb4_recon[i])
    recon_normal_unhyb_reg_perturb4 = Normalize()(unhyb_reg_prturb4_recon[i])
    recon_normal_hyb_base_perturb4 = Normalize()(hyb_base_prturb4_recon[i])
    recon_normal_hyb_reg_perturb4 = Normalize()(hyb_reg_prturb4_recon[i])

    recon_normal_unhyb_base_perturb3 = Normalize()(unhyb_base_prturb3_recon[i])
    recon_normal_unhyb_reg_perturb3 = Normalize()(unhyb_reg_prturb3_recon[i])
    recon_normal_hyb_base_perturb3 = Normalize()(hyb_base_prturb3_recon[i])
    recon_normal_hyb_reg_perturb3 = Normalize()(hyb_reg_prturb3_recon[i])

    recon_normal_unhyb_base_perturb2 = Normalize()(unhyb_base_prturb2_recon[i])
    recon_normal_unhyb_reg_perturb2 = Normalize()(unhyb_reg_prturb2_recon[i])
    recon_normal_hyb_base_perturb2 = Normalize()(hyb_base_prturb2_recon[i])
    recon_normal_hyb_reg_perturb2 = Normalize()(hyb_reg_prturb2_recon[i])

    recon_normal_unhyb_base_perturb1 = Normalize()(unhyb_base_prturb1_recon[i])
    recon_normal_unhyb_reg_perturb1 = Normalize()(unhyb_reg_prturb1_recon[i])
    recon_normal_hyb_base_perturb1 = Normalize()(hyb_base_prturb1_recon[i])
    recon_normal_hyb_reg_perturb1 = Normalize()(hyb_reg_prturb1_recon[i])

    ############################################################################################

    ssimlists_unhyb_base.append(max(ssim(testImage_normal[0], recon_normal_unhyb_base[0], data_range=1.), 0))
    ssimlists_unhyb_reg.append(max(ssim(testImage_normal[0], recon_normal_unhyb_reg[0], data_range=1.), 0))
    ssimlists_hyb_base.append(max(ssim(testImage_normal[0], recon_normal_hyb_base[0], data_range=1.), 0))
    ssimlists_hyb_reg.append(max(ssim(testImage_normal[0], recon_normal_hyb_reg[0], data_range=1.), 0))

    ssimlists_perturb4_base.append(max(ssim(testImage_normal[0], recon_normal_unhyb_base_perturb4[0], data_range=1.), 0))
    ssimlists_perturb4_reg.append(max(ssim(testImage_normal[0], recon_normal_unhyb_reg_perturb4[0], data_range=1.), 0))
    ssimlists_perturb4_hyb_base.append(max(ssim(testImage_normal[0], recon_normal_hyb_base_perturb4[0], data_range=1.), 0))
    ssimlists_perturb4_hyb_reg.append(max(ssim(testImage_normal[0], recon_normal_hyb_reg_perturb4[0], data_range=1.), 0))

    ssimlists_perturb3_base.append(max(ssim(testImage_normal[0], recon_normal_unhyb_base_perturb3[0], data_range=1.), 0))
    ssimlists_perturb3_reg.append(max(ssim(testImage_normal[0], recon_normal_unhyb_reg_perturb3[0], data_range=1.), 0))
    ssimlists_perturb3_hyb_base.append(max(ssim(testImage_normal[0], recon_normal_hyb_base_perturb3[0], data_range=1.), 0))
    ssimlists_perturb3_hyb_reg.append(max(ssim(testImage_normal[0], recon_normal_hyb_reg_perturb3[0], data_range=1.), 0))

    ssimlists_perturb2_base.append(max(ssim(testImage_normal[0], recon_normal_unhyb_base_perturb2[0], data_range=1.), 0))
    ssimlists_perturb2_reg.append(max(ssim(testImage_normal[0], recon_normal_unhyb_reg_perturb2[0], data_range=1.), 0))
    ssimlists_perturb2_hyb_base.append(max(ssim(testImage_normal[0], recon_normal_hyb_base_perturb2[0], data_range=1.), 0))
    ssimlists_perturb2_hyb_reg.append(max(ssim(testImage_normal[0], recon_normal_hyb_reg_perturb2[0], data_range=1.), 0))

    ssimlists_perturb1_base.append(max(ssim(testImage_normal[0], recon_normal_unhyb_base_perturb1[0], data_range=1.), 0))
    ssimlists_perturb1_reg.append(max(ssim(testImage_normal[0], recon_normal_unhyb_reg_perturb1[0], data_range=1.), 0))
    ssimlists_perturb1_hyb_base.append(max(ssim(testImage_normal[0], recon_normal_hyb_base_perturb1[0], data_range=1.), 0))
    ssimlists_perturb1_hyb_reg.append(max(ssim(testImage_normal[0], recon_normal_hyb_reg_perturb1[0], data_range=1.), 0))

    #############################################################################################

    psnrlists_unhyb_base.append(psnr(testImage_normal[0], recon_normal_unhyb_base[0], data_range=1.))
    psnrlists_unhyb_reg.append(psnr(testImage_normal[0], recon_normal_unhyb_reg[0], data_range=1.))
    psnrlists_hyb_base.append(psnr(testImage_normal[0], recon_normal_hyb_base[0], data_range=1.))
    psnrlists_hyb_reg.append(psnr(testImage_normal[0], recon_normal_hyb_reg[0], data_range=1.))

    psnrlists_perturb4_base.append(psnr(testImage_normal[0], recon_normal_unhyb_base_perturb4[0], data_range=1.))
    psnrlists_perturb4_reg.append(psnr(testImage_normal[0], recon_normal_unhyb_reg_perturb4[0], data_range=1.))
    psnrlists_perturb4_hyb_base.append(psnr(testImage_normal[0], recon_normal_hyb_base_perturb4[0], data_range=1.))
    psnrlists_perturb4_hyb_reg.append(psnr(testImage_normal[0], recon_normal_hyb_reg_perturb4[0], data_range=1.))

    psnrlists_perturb3_base.append(psnr(testImage_normal[0], recon_normal_unhyb_base_perturb3[0], data_range=1.))
    psnrlists_perturb3_reg.append(psnr(testImage_normal[0], recon_normal_unhyb_reg_perturb3[0], data_range=1.))
    psnrlists_perturb3_hyb_base.append(psnr(testImage_normal[0], recon_normal_hyb_base_perturb3[0], data_range=1.))
    psnrlists_perturb3_hyb_reg.append(psnr(testImage_normal[0], recon_normal_hyb_reg_perturb3[0], data_range=1.))

    psnrlists_perturb2_base.append(psnr(testImage_normal[0], recon_normal_unhyb_base_perturb2[0], data_range=1.))
    psnrlists_perturb2_reg.append(psnr(testImage_normal[0], recon_normal_unhyb_reg_perturb2[0], data_range=1.))
    psnrlists_perturb2_hyb_base.append(psnr(testImage_normal[0], recon_normal_hyb_base_perturb2[0], data_range=1.))
    psnrlists_perturb2_hyb_reg.append(psnr(testImage_normal[0], recon_normal_hyb_reg_perturb2[0], data_range=1.))

    psnrlists_perturb1_base.append(psnr(testImage_normal[0], recon_normal_unhyb_base_perturb1[0], data_range=1.))
    psnrlists_perturb1_reg.append(psnr(testImage_normal[0], recon_normal_unhyb_reg_perturb1[0], data_range=1.))
    psnrlists_perturb1_hyb_base.append(psnr(testImage_normal[0], recon_normal_hyb_base_perturb1[0], data_range=1.))
    psnrlists_perturb1_hyb_reg.append(psnr(testImage_normal[0], recon_normal_hyb_reg_perturb1[0], data_range=1.))


# reconstruction loss after training the model completely
loss_tre = torch.mean(((unhyb_base_prturb4_recon - testImages)**2)*0.5)

# reconstruction loss after training the model completely
loss_tre = torch.mean(((unhyb_reg_prturb4_recon - testImages)**2)*0.5)


import matplotlib.pyplot as plt
import numpy as np
print('SSIM of reconstruction on test data')
fracs = ['MLP-AE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
ssims = [ssimlists_unhyb_base, ssimlists_unhyb_reg, ssimlists_hyb_base, ssimlists_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('SSIM of reconstruction on test data')
ax1.boxplot(list(ssims))
#ax1.set_xlabel('Models', fontsize=16)
ax1.set_ylabel('SSIM', fontsize=15)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4], [str(s) for s in fracs], fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/mri_box_plots/SSIM_directReconOfMRITestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'RK_deg_'+str(deg_quad)+'.png')
plt.show()


#import matplotlib.pyplot as plt
#import numpy as np
print('PSNR of reconstruction on test data')
fracs = ['MLP-AE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
psnrs = [psnrlists_unhyb_base, psnrlists_unhyb_reg, psnrlists_hyb_base, psnrlists_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('PSNR of reconstruction on test data')
ax1.boxplot(list(psnrs))
#ax1.set_xlabel('Models', fontsize=16)
ax1.set_ylabel('PSNR(dB)', fontsize=15)
ax1.set_ylim([0,25])
plt.xticks([1, 2, 3, 4], [str(s) for s in fracs], fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/mri_box_plots/PSNR_directReconOfMRITestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'RK_deg_'+str(deg_quad)+'.png')
plt.show()

#Perturbation test for noise

print('PSNR of reconstruction of 70 % noised  test data')
fracs = ['MLP-AE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
psnrs = [psnrlists_perturb4_base, psnrlists_perturb4_reg, psnrlists_perturb4_hyb_base,psnrlists_perturb4_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('PSNR of reconstruction of 70 % noised  test data')
ax1.boxplot(list(psnrs))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('PSNR(dB)', fontsize=15)
ax1.set_ylim([0,25])
plt.xticks([1, 2, 3, 4], [str(s) for s in fracs], fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/mri_box_plots/PSNR_ReconOf70percentNoiseedMRITestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'RK_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(psnrlists_perturb4_base))
print(np.mean(psnrlists_perturb4_reg))
print(np.mean(psnrlists_perturb4_hyb_base))
print(np.mean(psnrlists_perturb4_hyb_reg))



#Perturbation test for noise
print('PSNR of reconstruction of 50 % noised  test data')
fracs = ['MLP-AE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
#fracs = ['MLP-AE','AE-REG', 'Hybrid MLP-AE', 'Hybrid AE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
psnrs = [psnrlists_perturb3_base, psnrlists_perturb3_reg, psnrlists_perturb3_hyb_base,psnrlists_perturb3_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('PSNR of reconstruction of 50 % noised  test data')
ax1.boxplot(list(psnrs))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('PSNR(dB)', fontsize=15)
ax1.set_ylim([0,25])
plt.xticks([1, 2, 3, 4], [str(s) for s in fracs], fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/mri_box_plots/PSNR_ReconOf50percentNoiseedMRITestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'RK_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(psnrlists_perturb3_base))
print(np.mean(psnrlists_perturb3_reg))
print(np.mean(psnrlists_perturb3_hyb_base))
print(np.mean(psnrlists_perturb3_hyb_reg))


#Perturbation test for noise
print('PSNR of reconstruction of 20 % noised  test data')
fracs = ['MLP-AE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
psnrs = [psnrlists_perturb2_base, psnrlists_perturb2_reg, psnrlists_perturb2_hyb_base,psnrlists_perturb2_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('PSNR of reconstruction of 20 % noised  test data')
ax1.boxplot(list(psnrs))
#ax1.set_xlabel('Models', fontsize=16)
ax1.set_ylabel('PSNR(dB)', fontsize=15)
ax1.set_ylim([0,25])
plt.xticks([1, 2, 3, 4], [str(s) for s in fracs], fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/mri_box_plots/PSNR_ReconOf20percentNoiseedMRITestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'RK_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(psnrlists_perturb2_base))
print(np.mean(psnrlists_perturb2_reg))
print(np.mean(psnrlists_perturb2_hyb_base))
print(np.mean(psnrlists_perturb2_hyb_reg))


#Perturbation test for noise
print('PSNR of reconstruction of 10 % noised  test data')
fracs = ['MLP-AE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
psnrs = [psnrlists_perturb1_base, psnrlists_perturb1_reg, psnrlists_perturb1_hyb_base,psnrlists_perturb1_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('PSNR of reconstruction of 10 % noised  test data')
ax1.boxplot(list(psnrs))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('PSNR(dB)', fontsize=15)
ax1.set_ylim([0,25])
plt.xticks([1, 2, 3, 4], [str(s) for s in fracs], fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/mri_box_plots/PSNR_ReconOf10percentNoiseedMRITestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'RK_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(psnrlists_perturb1_base))
print(np.mean(psnrlists_perturb1_reg))
print(np.mean(psnrlists_perturb1_hyb_base))
print(np.mean(psnrlists_perturb1_hyb_reg))


#Perturbation test for noise
print('SSIM of reconstruction of 70 % noised  test data')
fracs = ['MLP-AE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
ssims = [ssimlists_perturb4_base, ssimlists_perturb4_reg, ssimlists_perturb4_hyb_base, ssimlists_perturb4_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('SSIM of reconstruction of 70 % noised  test data')
ax1.boxplot(list(ssims))
#ax1.set_xlabel('Models', fontsize=16)
ax1.set_ylabel('SSIM', fontsize=15)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4], [str(s) for s in fracs], fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/mri_box_plots/SSIM_ReconOf70percentNoiseedMRITestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'RK_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(ssimlists_perturb4_base))
print(np.mean(ssimlists_perturb4_reg))
print(np.mean(ssimlists_perturb4_hyb_base))
print(np.mean(ssimlists_perturb4_hyb_reg))


#Perturbation test for noise

print('SSIM of reconstruction of 50 % noised  test data')
fracs = ['MLP-AE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
ssims = [ssimlists_perturb3_base, ssimlists_perturb3_reg, ssimlists_perturb3_hyb_base, ssimlists_perturb3_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('SSIM of reconstruction of 50 % noised  test data')
ax1.boxplot(list(ssims))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('SSIM', fontsize=15)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4], [str(s) for s in fracs], fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/mri_box_plots/SSIM_ReconOf50percentNoiseedMRITestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'RK_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(ssimlists_perturb3_base))
print(np.mean(ssimlists_perturb3_reg))
print(np.mean(ssimlists_perturb3_hyb_base))
print(np.mean(ssimlists_perturb3_hyb_reg))


#Perturbation test for noise
print('SSIM of reconstruction of 20 % noised  test data')
fracs = ['MLP-AE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
ssims = [ssimlists_perturb2_base, ssimlists_perturb2_reg, ssimlists_perturb2_hyb_base, ssimlists_perturb2_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('SSIM of reconstruction of 20 % noised  test data')
ax1.boxplot(list(ssims))
#ax1.set_xlabel('Models', fontsize=15)
ax1.set_ylabel('SSIM', fontsize=15)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4], [str(s) for s in fracs], fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/mri_box_plots/SSIM_ReconOf20percentNoiseedMRITestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'RK_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(ssimlists_perturb2_base))
print(np.mean(ssimlists_perturb2_reg))
print(np.mean(ssimlists_perturb2_hyb_base))
print(np.mean(ssimlists_perturb2_hyb_reg))


#Perturbation test for noise
print('SSIM of reconstruction of 10 % noised  test data')
fracs = ['MLP-AE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
ssims = [ssimlists_perturb1_base, ssimlists_perturb1_reg, ssimlists_perturb1_hyb_base, ssimlists_perturb1_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('SSIM of reconstruction of 10 % noised  test data')
ax1.boxplot(list(ssims))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('SSIM', fontsize=15)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4], [str(s) for s in fracs], fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/mri_box_plots/SSIM_ReconOf10percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'RK_deg_'+str(deg_quad)+'.png')
plt.show()


print(np.mean(ssimlists_perturb1_base))
print(np.mean(ssimlists_perturb1_reg))
print(np.mean(ssimlists_perturb1_hyb_base))
print(np.mean(ssimlists_perturb1_hyb_reg))