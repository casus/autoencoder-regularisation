import sys
sys.path.append('./')


from get_data import get_data, get_data_train, get_data_val
import torch
import os
import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline

from datasets import InMemDataLoader
import torch.nn.functional as F
import torch
from torch import nn

import nibabel as nib     # Read / write access to some common neuroimaging file formats
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
from scipy import interpolate
import ot

import jmp_solver1.surrogates
import matplotlib
matplotlib.rcdefaults() 

import scipy
import scipy.integrate
from vae_models_for_fmnist import VAE_try_MRI, VAE_mlp_MRI


latent_dim = 40
frac = 0.2   # Training data amount



path_in_repo = './models_saved/'

Hybrid_poly_deg = 80

deg_quad = Hybrid_poly_deg

u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
x = np.linspace(-1,1,96)
X_p = u_ob.data_axes([x,x]).T

def get_all_thetas(listedImage):
    #print('listedImage.shape',listedImage.shape)
    Fr = torch.tensor(listedImage).reshape(96*96)

    def grad_x(t,theta):
        theta_t = torch.tensor(theta)
        return -2*torch.matmul(X_p.T,(torch.matmul(X_p,theta_t)-Fr)).detach().numpy()

    def give_theta_t():
        #start = time.time()
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

    return act_theta


#before executing this cell you need to have preexisting trained coefficients for what ever degree you want
trainImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/trainDataSet.pt',map_location=torch.device('cpu'))
trainCoeffs = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/trainDataRK_coeffs.pt',map_location=torch.device('cpu'))


testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/testDataSet.pt',map_location=torch.device('cpu'))
testCoeffs = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/testDataRK_coeffs.pt',map_location=torch.device('cpu'))

perturbed_testImages_noise_percent_10 = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/MRI_testImages_N_10.pt',map_location=torch.device('cpu'))
perturbed_testImages_noise_percent_20 = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/MRI_testImages_N_20.pt',map_location=torch.device('cpu'))
perturbed_testImages_noise_percent_50 = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/MRI_testImages_N_50.pt',map_location=torch.device('cpu'))
perturbed_testImages_noise_percent_70 = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/MRI_testImages_N_70.pt',map_location=torch.device('cpu'))


print('perturbed_testImages_noise_percent_10.shape', perturbed_testImages_noise_percent_10.shape)


perturbed_testCoeffs_10 = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/coeffs_saved/LSTSQ_N10_testcoeffs_MRI_dq'+str(deg_quad)+'.pt',map_location=torch.device('cpu'))
perturbed_testCoeffs_20 = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/coeffs_saved/LSTSQ_N20_testcoeffs_MRI_dq'+str(deg_quad)+'.pt',map_location=torch.device('cpu'))
perturbed_testCoeffs_50 = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/coeffs_saved/LSTSQ_N50_testcoeffs_MRI_dq'+str(deg_quad)+'.pt',map_location=torch.device('cpu'))
perturbed_testCoeffs_70 = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/coeffs_saved/LSTSQ_N70_testcoeffs_MRI_dq'+str(deg_quad)+'.pt',map_location=torch.device('cpu'))


Analys_size = 200

trainImages = trainImages[:Analys_size]
trainCoeffs = trainCoeffs[:Analys_size]

testImages = testImages[:Analys_size]
testCoeffs = testCoeffs[:Analys_size]


from models import AE
from activations import Sin


reg_nodes_sampling = 'legendre'
alpha = 0.5
alpha_reg = 0.5
frac_reg_no_reg = frac
hidden_size = 1000
deg_poly = 21
deg_poly_forRK = 21


lr = 0.0001
no_layers = 5
no_reg_samples = 10

name_hyb = '_'+reg_nodes_sampling+'_'+str(frac_reg_no_reg)+'_'+str(alpha_reg)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)
name_unhyb = '_'+reg_nodes_sampling+'_'+'_'+str(frac_reg_no_reg)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)+'_'+str(no_reg_samples)#+'_'+str(no_epochs)
      
inp_dim_hyb = (deg_quad+1)*(deg_quad+1)

inp_dim_unhyb = [1,96,96]

model_reg = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
model_base = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

RK_model_reg = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
RK_model_base = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

'''#if(frac == 0.9):
    name_hyb_ = '_'+reg_nodes_sampling+'_'+str(frac)+'_'+str(alpha_full)+'_'+str(hidden_size)+'_'+str(deg_poly_full)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)
    name_unhyb_ = '_'+reg_nodes_sampling+'_'+str(frac)+'_'+str(alpha_full)+'_'+str(hidden_size)+'_'+str(deg_poly_full)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

    path_hyb = path_in_repo

    RK_model_reg.load_state_dict(torch.load(path_hyb+'model_reg'+name_hyb_, map_location=torch.device('cpu')))
    RK_model_base.load_state_dict(torch.load(path_hyb+'model_base'+name_hyb_, map_location=torch.device('cpu')))

    torch.save(RK_model_reg.state_dict(), path_in_repo+'/model_reg_RKMRI_TDA'+name_hyb)
    torch.save(RK_model_base.state_dict(), path_in_repo+'/model_base_RKMRI_TDA'+name_hyb) 

    path_unhyb = path_in_repo

    model_reg.load_state_dict(torch.load(path_unhyb+'model_reg'+name_unhyb_, map_location=torch.device('cpu')))
    model_base.load_state_dict(torch.load(path_unhyb+'model_base'+name_unhyb_, map_location=torch.device('cpu')))

    torch.save(model_reg.state_dict(), path_in_repo+'/model_reg_MRI_TDA'+name_unhyb) 
    torch.save(model_reg.state_dict(), path_in_repo+'/model_base_MRI_TDA'+name_unhyb) 


#else:'''

path_hyb = path_in_repo

RK_model_reg.load_state_dict(torch.load(path_hyb+'model_reg_RKMRI_TDA'+name_hyb, map_location=torch.device('cuda')))
RK_model_base.load_state_dict(torch.load(path_hyb+'model_base_RKMRI_TDA'+name_hyb, map_location=torch.device('cuda')))



path_unhyb_new = path_in_repo

model_reg.load_state_dict(torch.load(path_unhyb_new+'model_reg_MRI_TDA'+name_unhyb, map_location=torch.device('cuda')))
model_base.load_state_dict(torch.load(path_unhyb_new+'model_base_MRI_TDA'+name_unhyb, map_location=torch.device('cuda')))




#/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/model_reg_MRI_TDA_legendre__0.4_0.5_1000_21_70_0.0001_5_10
#model_reg.eval()
#model_base.eval()

print("Anything here to see")
#from vae import BetaVAE
from activations import Sin
activation = Sin()
from vae_models_for_fmnist import VAE_try, VAE_mlp, Autoencoder_linear_MRI

#loading convolutional autoencoder
from convAE import ConvoAE_MRI
no_layers_cae = 5
latent_dim_cae = latent_dim
lr_cae =1e-4
if(frac == 0.8):
    frac = 1.0

frac_cae = frac

#path_cae_mri = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'

path_cae_mri = path_in_repo

name_unhyb_cae = '_'+str(frac_cae)+'_'+str(latent_dim_cae)+'_'+str(lr_cae)+'_'+str(no_layers_cae)
model_convAE = ConvoAE_MRI(latent_dim_cae).to(device)
model_convAE.load_state_dict(torch.load(path_cae_mri+'model_base_cae_MRI'+name_unhyb_cae, map_location=torch.device(device)), strict=False)



print("loaded ?")
#path_unhyb_bvae = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'
path_unhyb_bvae = path_in_repo

TDA_bvae = frac
alpha_bvae = 0.1
deg_poly_bvae = 21
latent_dim_bvae = latent_dim
no_layers_bvae = 5
hidden_size = 1000
name_bvae = '_'+reg_nodes_sampling+'_'+'_'+str(TDA_bvae)+'_'+str(alpha_bvae)+'_'+str(hidden_size)+'_'+str(deg_poly_bvae)+'_'+str(latent_dim_bvae)+'_'+str(lr)+'_'+str(no_layers_bvae)
#from vae import BetaVAE
#model_betaVAE = BetaVAE(batch_size = 1, img_depth = 1, net_depth = no_layers, z_dim = latent_dim_bvae, img_dim = 96).to(device)
#model_betaVAE.load_state_dict(torch.load(path_unhyb_bvae+'model_base_bvae_MRI'+name_bvae, map_location=torch.device(device)), strict=False)


from activations import Sin
activation = Sin()
#model_mlpVAE = MLPVAE(1*96*96, hidden_size, latent_dim, 
                #    no_layers, activation).to(device) # regularised autoencoder
model_mlpVAE_ = VAE_mlp_MRI(image_size=96*96, h_dim=1000, z_dim=latent_dim).to(device)

#model_betaVAE = BetaVAE([96, 96], 1, no_filters=4, no_layers=3,
                #kernel_size=3, latent_dim=10, activation = Sin()).to(device) # regularised autoencoder

model_cnnVAE_ = VAE_try_MRI(image_channels=1, h_dim=16*9*9, z_dim=latent_dim).to(device)

#model_betaVAE = BetaVAE(batch_size = 1, img_depth = 1, net_depth = no_layers, z_dim = latent_dim, img_dim = 96).to(device)
model_mlpVAE_.load_state_dict(torch.load(path_unhyb_bvae+'model_base_mlp_vae_MRI'+name_bvae, map_location=torch.device(device)), strict=False)
model_cnnVAE_.load_state_dict(torch.load(path_unhyb_bvae+'model_base_cnn_vae_MRI'+name_bvae, map_location=torch.device(device)), strict=False)


print("loaded bvae?")

#loading contractive autoencoder
no_layers_contraae = 5
latent_dim_contraae = latent_dim
lr_contraae =1e-3
frac_contraae = frac
#path_unhyb_contraae = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'
path_unhyb_contraae = path_in_repo

name_unhyb_contraae = '_'+str(frac_contraae)+'_'+str(latent_dim_contraae)+'_'+str(lr_contraae)+'_'+str(no_layers_contraae)
model_contra_ = Autoencoder_linear_MRI(latent_dim_contraae).to(device)
model_contra_.load_state_dict(torch.load(path_unhyb_contraae+'model_base_contraAE_MRI'+name_unhyb_contraae, map_location=torch.device(device)), strict=False)


#############################################################################################################################################################################


#rec = model_convAE(trainImages).view(trainImages.shape).detach().numpy() 

#from vae import BetaVAE
'''from activations import Sin
activation = Sin()
from vae_models_for_fmnist import VAE_try, VAE_mlp, Autoencoder_linear
#model_mlpVAE = MLPVAE(1*96*96, hidden_size, latent_dim, 
                #    no_layers, activation).to(device) # regularised autoencoder

model_mlpVAE_ = VAE_mlp(96*96, hidden_size, latent_dim).to(device)

#model_betaVAE = BetaVAE([96, 96], 1, no_filters=4, no_layers=3,
                #kernel_size=3, latent_dim=10, activation = Sin()).to(device) # regularised autoencoder

model_cnnVAE_ = VAE_try(image_channels=1, h_dim=8*2*2, z_dim=latent_dim).to(device)

#model_betaVAE = BetaVAE(batch_size = 1, img_depth = 1, net_depth = no_layers, z_dim = latent_dim, img_dim = 96).to(device)
model_mlpVAE_.load_state_dict(torch.load(path_unhyb+'model_base_mlp_vae_TDA'+name_unhyb, map_location=torch.device(device)), strict=False)
model_cnnVAE_.load_state_dict(torch.load(path_unhyb+'model_base_cnn_vae_TDA'+name_unhyb, map_location=torch.device(device)), strict=False)'''

def model_mlpVAE(input):
    #print('model_betaVAE(input).shape', model_betaVAE(input).shape)
    input = input.reshape(-1, 96*96)
    recon,_,_= model_mlpVAE_(input)
    return recon

def model_cnnVAE(input):
    #print('model_betaVAE(input).shape', model_betaVAE(input).shape)
    recon,_,_= model_cnnVAE_(input)
    return recon

'''def model_contra(input):
    input = input.reshape(-1, 96*96)
    recon,_,_= model_cnnVAE_(input)
    return recon'''


def model_contra(input):
    input = input.reshape(-1, 96*96)
    recon= model_contra_(input)
    return recon
#rec = model_convAE(torch.from_numpy(trainImages).reshape(1,1,96,96).to(device))

#loading contractive autoencoder
'''no_layers_contraae = 3
latent_dim_contraae = latent_dim
lr_contraae =1e-3
name_unhyb_contraae = '_'+str(frac)+'_'+str(latent_dim_contraae)+'_'+str(lr_contraae)+'_'+str(no_layers_contraae)
model_contra_ = Autoencoder_linear(latent_dim).to(device)
model_contra_.load_state_dict(torch.load(path_unhyb+'model_base_contraAE_TDA'+name_unhyb_contraae, map_location=torch.device(device)), strict=False)'''

def wass(original, recon):
    wassDistance_sliced = ot.sliced_wasserstein_distance(torch.tensor(original), torch.tensor(recon), seed=0,)  
    return wassDistance_sliced

print()
print("anything ? all ok")


unhyb_rec_rAE_train = model_reg(trainImages).view(trainImages.shape).detach().numpy() 
unhyb_rec_bAE_train = model_base(trainImages).view(trainImages.shape).detach().numpy() 
unhyb_rec_rAE_train = torch.tensor(unhyb_rec_rAE_train, requires_grad=False)
unhyb_rec_bAE_train = torch.tensor(unhyb_rec_bAE_train, requires_grad=False)


unhyb_rec_rAE_test = model_reg(testImages).view(testImages.shape).detach().numpy() 
unhyb_rec_bAE_test = model_base(testImages).view(testImages.shape).detach().numpy() 
convAE_rec_test = model_convAE(testImages).view(testImages.shape).detach().numpy() 
mlpVAE_rec_test = model_mlpVAE(testImages).view(testImages.shape).detach().numpy() 
cnnVAE_rec_test = model_cnnVAE(testImages).view(testImages.shape).detach().numpy() 
contra_rec_test = model_contra(testImages).view(testImages.shape).detach().numpy() 


unhyb_rec_rAE_test = torch.tensor(unhyb_rec_rAE_test, requires_grad=False)
unhyb_rec_bAE_test = torch.tensor(unhyb_rec_bAE_test, requires_grad=False)
convAE_rec_test = torch.tensor(convAE_rec_test, requires_grad=False)
mlpVAE_rec_test = torch.tensor(mlpVAE_rec_test, requires_grad=False)
cnnVAE_rec_test = torch.tensor(cnnVAE_rec_test, requires_grad=False)
contra_rec_test = torch.tensor(contra_rec_test, requires_grad=False)


rec_rAE_train = RK_model_reg(trainCoeffs.float()).view(trainCoeffs.shape)
rec_bAE_train = RK_model_base(trainCoeffs.float()).view(trainCoeffs.shape)

rec_rAE_test = RK_model_reg(testCoeffs.float()).view(testCoeffs.shape)
rec_bAE_test = RK_model_base(testCoeffs.float()).view(testCoeffs.shape)




# reconstruction loss after training the model completely
loss_tre = torch.mean(((unhyb_rec_rAE_train - testImages)**2)*0.5)


# reconstruction loss after training the model completely
loss_tre = torch.mean(((unhyb_rec_bAE_train - testImages)**2)*0.5)


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




#unhyb_base_prturb4_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/all_perturb4_recons_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'.pt')
#unhyb_base_prturb4_recon = unhyb_base_prturb4_recon.reshape(200, 1, 96, 96)
#unhyb_base_prturb4_recon = torch.tensor(unhyb_base_prturb4_recon, requires_grad=False)
unhyb_base_prturb4_recon = model_base(perturbed_testImages_noise_percent_70.float()).view(perturbed_testImages_noise_percent_70.shape).detach().numpy() 
unhyb_base_prturb4_recon = torch.tensor(unhyb_base_prturb4_recon, requires_grad=False)
unhyb_base_prturb4_recon = unhyb_base_prturb4_recon.reshape(200, 1, 96, 96)

#here convAE
convAE_prturb4_recon = model_convAE(perturbed_testImages_noise_percent_70.float()).view(perturbed_testImages_noise_percent_70.shape).detach().numpy() 
convAE_prturb4_recon = torch.tensor(convAE_prturb4_recon, requires_grad=False)
convAE_prturb4_recon = convAE_prturb4_recon.reshape(200, 1, 96, 96)

#mlpVAE here
mlpVAE_prturb4_recon = model_mlpVAE(perturbed_testImages_noise_percent_70.float()).view(perturbed_testImages_noise_percent_70.shape).detach().numpy() 
mlpVAE_prturb4_recon = torch.tensor(mlpVAE_prturb4_recon, requires_grad=False)
mlpVAE_prturb4_recon = mlpVAE_prturb4_recon.reshape(200, 1, 96, 96)


#cnnVAE here
cnnVAE_prturb4_recon = model_cnnVAE(perturbed_testImages_noise_percent_70.float()).view(perturbed_testImages_noise_percent_70.shape).detach().numpy() 
cnnVAE_prturb4_recon = torch.tensor(cnnVAE_prturb4_recon, requires_grad=False)
cnnVAE_prturb4_recon = cnnVAE_prturb4_recon.reshape(200, 1, 96, 96)

#contra here
contra_prturb4_recon = model_contra(perturbed_testImages_noise_percent_70.float()).view(perturbed_testImages_noise_percent_70.shape).detach().numpy() 
contra_prturb4_recon = torch.tensor(contra_prturb4_recon, requires_grad=False)
contra_prturb4_recon = contra_prturb4_recon.reshape(200, 1, 96, 96)

#unhyb_reg_prturb4_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/all_perturb4_recons_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'.pt') 
#unhyb_reg_prturb4_recon = unhyb_reg_prturb4_recon.reshape(200, 1, 96, 96)
#unhyb_reg_prturb4_recon = torch.tensor(unhyb_reg_prturb4_recon, requires_grad=False)

unhyb_reg_prturb4_recon = model_reg(perturbed_testImages_noise_percent_70.float()).view(perturbed_testImages_noise_percent_70.shape).detach().numpy() 
unhyb_reg_prturb4_recon = torch.tensor(unhyb_reg_prturb4_recon, requires_grad=False)
unhyb_reg_prturb4_recon = unhyb_reg_prturb4_recon.reshape(200, 1, 96, 96)

#hyb_base_prturb4_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/LSTQS'+str(deg_quad)+'_all_perturb4_recons_RK_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
#hyb_base_prturb4_recon = hyb_base_prturb4_recon.reshape(200, 1, 96, 96)
#hyb_base_prturb4_recon = torch.tensor(hyb_base_prturb4_recon, requires_grad=False).


hyb_base_prturb4_recon = RK_model_base(perturbed_testCoeffs_70.float()).view(perturbed_testCoeffs_70.shape)
hyb_base_prturb4_recon = torch.tensor(hyb_base_prturb4_recon, requires_grad=False)
hyb_base_prturb4_recon = torch.matmul(X_p.float(), hyb_base_prturb4_recon.squeeze(1).T).T
hyb_base_prturb4_recon = hyb_base_prturb4_recon.reshape(Analys_size,1,96,96)

#hyb_reg_prturb4_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/LSTQS'+str(deg_quad)+'_all_perturb4_recons_RK_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
#hyb_reg_prturb4_recon = hyb_reg_prturb4_recon.reshape(200, 1, 96, 96)
#hyb_reg_prturb4_recon = torch.tensor(hyb_reg_prturb4_recon, requires_grad=False)

hyb_reg_prturb4_recon = RK_model_reg(perturbed_testCoeffs_70.float()).view(perturbed_testCoeffs_70.shape)
hyb_reg_prturb4_recon = torch.tensor(hyb_reg_prturb4_recon, requires_grad=False)
hyb_reg_prturb4_recon = torch.matmul(X_p.float(), hyb_reg_prturb4_recon.squeeze(1).T).T
hyb_reg_prturb4_recon = hyb_reg_prturb4_recon.reshape(Analys_size,1,96,96)


#unhyb_base_prturb3_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/all_perturb3_recons_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'.pt')
#unhyb_base_prturb3_recon = unhyb_base_prturb3_recon.reshape(200, 1, 96, 96)
#unhyb_base_prturb3_recon = torch.tensor(unhyb_base_prturb3_recon, requires_grad=False)

unhyb_base_prturb3_recon = model_base(perturbed_testImages_noise_percent_50.float()).view(perturbed_testImages_noise_percent_50.shape).detach().numpy() 
unhyb_base_prturb3_recon = torch.tensor(unhyb_base_prturb3_recon, requires_grad=False)
unhyb_base_prturb3_recon = unhyb_base_prturb3_recon.reshape(200, 1, 96, 96)

#here convAE
convAE_prturb3_recon = model_convAE(perturbed_testImages_noise_percent_50.float()).view(perturbed_testImages_noise_percent_50.shape).detach().numpy() 
convAE_prturb3_recon = torch.tensor(convAE_prturb3_recon, requires_grad=False)
convAE_prturb3_recon = convAE_prturb3_recon.reshape(200, 1, 96, 96)

# MLP VAE
mlpVAE_prturb3_recon = model_mlpVAE(perturbed_testImages_noise_percent_50.float()).view(perturbed_testImages_noise_percent_50.shape).detach().numpy() 
mlpVAE_prturb3_recon = torch.tensor(mlpVAE_prturb3_recon, requires_grad=False)
mlpVAE_prturb3_recon = mlpVAE_prturb3_recon.reshape(200, 1, 96, 96)

# CNN VAE
cnnVAE_prturb3_recon = model_cnnVAE(perturbed_testImages_noise_percent_50.float()).view(perturbed_testImages_noise_percent_50.shape).detach().numpy() 
cnnVAE_prturb3_recon = torch.tensor(cnnVAE_prturb3_recon, requires_grad=False)
cnnVAE_prturb3_recon = cnnVAE_prturb3_recon.reshape(200, 1, 96, 96)

# contra VAE
contra_prturb3_recon = model_contra(perturbed_testImages_noise_percent_50.float()).view(perturbed_testImages_noise_percent_50.shape).detach().numpy() 
contra_prturb3_recon = torch.tensor(contra_prturb3_recon, requires_grad=False)
contra_prturb3_recon = contra_prturb3_recon.reshape(200, 1, 96, 96)


#unhyb_reg_prturb3_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/all_perturb3_recons_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'.pt')
#unhyb_reg_prturb3_recon = unhyb_reg_prturb3_recon.reshape(200, 1, 96, 96)
#unhyb_reg_prturb3_recon = torch.tensor(unhyb_reg_prturb3_recon, requires_grad=False)

unhyb_reg_prturb3_recon = model_reg(perturbed_testImages_noise_percent_50.float()).view(perturbed_testImages_noise_percent_50.shape).detach().numpy() 
unhyb_reg_prturb3_recon = torch.tensor(unhyb_reg_prturb3_recon, requires_grad=False)
unhyb_reg_prturb3_recon = unhyb_reg_prturb3_recon.reshape(200, 1, 96, 96)


#hyb_base_prturb3_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/LSTQS'+str(deg_quad)+'_all_perturb3_recons_RK_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
#hyb_base_prturb3_recon = hyb_base_prturb3_recon.reshape(200, 1, 96, 96)
#hyb_base_prturb3_recon = torch.tensor(hyb_base_prturb3_recon, requires_grad=False)

hyb_base_prturb3_recon = RK_model_base(perturbed_testCoeffs_50.float()).view(perturbed_testCoeffs_50.shape)
hyb_base_prturb3_recon = torch.tensor(hyb_base_prturb3_recon, requires_grad=False)
hyb_base_prturb3_recon = torch.matmul(X_p.float(), hyb_base_prturb3_recon.squeeze(1).T).T
hyb_base_prturb3_recon = hyb_base_prturb3_recon.reshape(Analys_size,1,96,96)

#hyb_reg_prturb3_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/LSTQS'+str(deg_quad)+'_all_perturb3_recons_RK_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
#hyb_reg_prturb3_recon = hyb_reg_prturb3_recon.reshape(200, 1, 96, 96)
#hyb_reg_prturb3_recon = torch.tensor(hyb_reg_prturb3_recon, requires_grad=False)

hyb_reg_prturb3_recon = RK_model_reg(perturbed_testCoeffs_50.float()).view(perturbed_testCoeffs_50.shape)
hyb_reg_prturb3_recon = torch.tensor(hyb_reg_prturb3_recon, requires_grad=False)
hyb_reg_prturb3_recon = torch.matmul(X_p.float(), hyb_reg_prturb3_recon.squeeze(1).T).T
hyb_reg_prturb3_recon = hyb_reg_prturb3_recon.reshape(Analys_size,1,96,96)


#unhyb_base_prturb2_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/all_perturb2_recons_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'.pt')
#unhyb_base_prturb2_recon = unhyb_base_prturb2_recon.reshape(200, 1, 96, 96)
#unhyb_base_prturb2_recon = torch.tensor(unhyb_base_prturb2_recon, requires_grad=False)

unhyb_base_prturb2_recon = model_base(perturbed_testImages_noise_percent_20.float()).view(perturbed_testImages_noise_percent_20.shape).detach().numpy() 
unhyb_base_prturb2_recon = torch.tensor(unhyb_base_prturb2_recon, requires_grad=False)
unhyb_base_prturb2_recon = unhyb_base_prturb2_recon.reshape(200, 1, 96, 96)

#convAE here
convAE_prturb2_recon = model_convAE(perturbed_testImages_noise_percent_20.float()).view(perturbed_testImages_noise_percent_20.shape).detach().numpy() 
convAE_prturb2_recon = torch.tensor(convAE_prturb2_recon, requires_grad=False)
convAE_prturb2_recon = convAE_prturb2_recon.reshape(200, 1, 96, 96)

#mlpVAE here
mlpVAE_prturb2_recon = model_mlpVAE(perturbed_testImages_noise_percent_20.float()).view(perturbed_testImages_noise_percent_20.shape).detach().numpy() 
mlpVAE_prturb2_recon = torch.tensor(mlpVAE_prturb2_recon, requires_grad=False)
mlpVAE_prturb2_recon = mlpVAE_prturb2_recon.reshape(200, 1, 96, 96)

#cnnVAE here
cnnVAE_prturb2_recon = model_cnnVAE(perturbed_testImages_noise_percent_20.float()).view(perturbed_testImages_noise_percent_20.shape).detach().numpy() 
cnnVAE_prturb2_recon = torch.tensor(cnnVAE_prturb2_recon, requires_grad=False)
cnnVAE_prturb2_recon = cnnVAE_prturb2_recon.reshape(200, 1, 96, 96)

#contraVAE here
contra_prturb2_recon = model_contra(perturbed_testImages_noise_percent_20.float()).view(perturbed_testImages_noise_percent_20.shape).detach().numpy() 
contra_prturb2_recon = torch.tensor(contra_prturb2_recon, requires_grad=False)
contra_prturb2_recon = contra_prturb2_recon.reshape(200, 1, 96, 96)

#unhyb_reg_prturb2_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/all_perturb2_recons_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'.pt')
#unhyb_reg_prturb2_recon = unhyb_reg_prturb2_recon.reshape(200, 1, 96, 96)
#unhyb_reg_prturb2_recon = torch.tensor(unhyb_reg_prturb2_recon, requires_grad=False)

unhyb_reg_prturb2_recon = model_reg(perturbed_testImages_noise_percent_20.float()).view(perturbed_testImages_noise_percent_20.shape).detach().numpy() 
unhyb_reg_prturb2_recon = torch.tensor(unhyb_reg_prturb2_recon, requires_grad=False)
unhyb_reg_prturb2_recon = unhyb_reg_prturb2_recon.reshape(200, 1, 96, 96)


#hyb_base_prturb2_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/LSTQS'+str(deg_quad)+'_all_perturb2_recons_RK_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
#hyb_base_prturb2_recon = hyb_base_prturb2_recon.reshape(200, 1, 96, 96)
#hyb_base_prturb2_recon = torch.tensor(hyb_base_prturb2_recon, requires_grad=False)

hyb_base_prturb2_recon = RK_model_base(perturbed_testCoeffs_20.float()).view(perturbed_testCoeffs_20.shape)
hyb_base_prturb2_recon = torch.tensor(hyb_base_prturb2_recon, requires_grad=False)
hyb_base_prturb2_recon = torch.matmul(X_p.float(), hyb_base_prturb2_recon.squeeze(1).T).T
hyb_base_prturb2_recon = hyb_base_prturb2_recon.reshape(Analys_size,1,96,96)

#hyb_reg_prturb2_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/LSTQS'+str(deg_quad)+'_all_perturb2_recons_RK_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
#hyb_reg_prturb2_recon = hyb_reg_prturb2_recon.reshape(200, 1, 96, 96)
#hyb_reg_prturb2_recon = torch.tensor(hyb_reg_prturb2_recon, requires_grad=False)

hyb_reg_prturb2_recon = RK_model_reg(perturbed_testCoeffs_20.float()).view(perturbed_testCoeffs_20.shape)
hyb_reg_prturb2_recon = torch.tensor(hyb_reg_prturb2_recon, requires_grad=False)
hyb_reg_prturb2_recon = torch.matmul(X_p.float(), hyb_reg_prturb2_recon.squeeze(1).T).T
hyb_reg_prturb2_recon = hyb_reg_prturb2_recon.reshape(Analys_size,1,96,96)

#unhyb_base_prturb1_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/all_perturb1_recons_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'.pt')
#unhyb_base_prturb1_recon = unhyb_base_prturb1_recon.reshape(200, 1, 96, 96)
#unhyb_base_prturb1_recon = torch.tensor(unhyb_base_prturb1_recon, requires_grad=False)

unhyb_base_prturb1_recon = model_base(perturbed_testImages_noise_percent_10.float()).view(perturbed_testImages_noise_percent_10.shape).detach().numpy() 
unhyb_base_prturb1_recon = torch.tensor(unhyb_base_prturb1_recon, requires_grad=False)
unhyb_base_prturb1_recon = unhyb_base_prturb1_recon.reshape(200, 1, 96, 96)

#convAE Here
convAE_prturb1_recon = model_convAE(perturbed_testImages_noise_percent_10.float()).view(perturbed_testImages_noise_percent_10.shape).detach().numpy() 
convAE_prturb1_recon = torch.tensor(convAE_prturb1_recon, requires_grad=False)
convAE_prturb1_recon = convAE_prturb1_recon.reshape(200, 1, 96, 96)

#mlpVAE Here
mlpVAE_prturb1_recon = model_mlpVAE(perturbed_testImages_noise_percent_10.float()).view(perturbed_testImages_noise_percent_10.shape).detach().numpy() 
mlpVAE_prturb1_recon = torch.tensor(mlpVAE_prturb1_recon, requires_grad=False)
mlpVAE_prturb1_recon = mlpVAE_prturb1_recon.reshape(200, 1, 96, 96)

#cnnVAE Here
cnnVAE_prturb1_recon = model_cnnVAE(perturbed_testImages_noise_percent_10.float()).view(perturbed_testImages_noise_percent_10.shape).detach().numpy() 
cnnVAE_prturb1_recon = torch.tensor(cnnVAE_prturb1_recon, requires_grad=False)
cnnVAE_prturb1_recon = cnnVAE_prturb1_recon.reshape(200, 1, 96, 96)

#contraVAE Here
contra_prturb1_recon = model_contra(perturbed_testImages_noise_percent_10.float()).view(perturbed_testImages_noise_percent_10.shape).detach().numpy() 
contra_prturb1_recon = torch.tensor(contra_prturb1_recon, requires_grad=False)
contra_prturb1_recon = contra_prturb1_recon.reshape(200, 1, 96, 96)

#unhyb_reg_prturb1_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/all_perturb1_recons_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'.pt')
#unhyb_reg_prturb1_recon = unhyb_reg_prturb1_recon.reshape(200, 1, 96, 96)
#unhyb_reg_prturb1_recon = torch.tensor(unhyb_reg_prturb1_recon, requires_grad=False)

unhyb_reg_prturb1_recon = model_reg(perturbed_testImages_noise_percent_10.float()).view(perturbed_testImages_noise_percent_10.shape).detach().numpy() 
unhyb_reg_prturb1_recon = torch.tensor(unhyb_reg_prturb1_recon, requires_grad=False)
unhyb_reg_prturb1_recon = unhyb_reg_prturb1_recon.reshape(200, 1, 96, 96)

#hyb_base_prturb1_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/LSTQS'+str(deg_quad)+'_all_perturb1_recons_RK_baseline_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
#hyb_base_prturb1_recon = hyb_base_prturb1_recon.reshape(200, 1, 96, 96)
#hyb_base_prturb1_recon = torch.tensor(hyb_base_prturb1_recon, requires_grad=False)

hyb_base_prturb1_recon = RK_model_base(perturbed_testCoeffs_10.float()).view(perturbed_testCoeffs_10.shape)
hyb_base_prturb1_recon = torch.tensor(hyb_base_prturb1_recon, requires_grad=False)
hyb_base_prturb1_recon = torch.matmul(X_p.float(), hyb_base_prturb1_recon.squeeze(1).T).T
hyb_base_prturb1_recon = hyb_base_prturb1_recon.reshape(Analys_size,1,96,96)

#hyb_reg_prturb1_recon = torch.load('/home/ramana44/autoencoder_regulrization_conf_tasks/perturbedReconstructions/LSTQS'+str(deg_quad)+'_all_perturb1_recons_RK_regularized_Lat'+str(latent_dim)+'_TDA'+str(frac)+'lot.pt')
#hyb_reg_prturb1_recon = hyb_reg_prturb1_recon.reshape(200, 1, 96, 96)
#hyb_reg_prturb1_recon = torch.tensor(hyb_reg_prturb1_recon, requires_grad=False)

hyb_reg_prturb1_recon = RK_model_reg(perturbed_testCoeffs_10.float()).view(perturbed_testCoeffs_10.shape)
hyb_reg_prturb1_recon = torch.tensor(hyb_reg_prturb1_recon, requires_grad=False)
hyb_reg_prturb1_recon = torch.matmul(X_p.float(), hyb_reg_prturb1_recon.squeeze(1).T).T
hyb_reg_prturb1_recon = hyb_reg_prturb1_recon.reshape(Analys_size,1,96,96)


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

####################################################################################################

wasslists_unhyb_base = []
wasslists_convAE = []
wasslists_mlpVAE = []
wasslists_cnnVAE = []
wasslists_contra = []
wasslists_unhyb_reg = []
wasslists_hyb_base = []
wasslists_hyb_reg = []

wasslists_perturb4_hyb_base = []
wasslists_perturb4_hyb_reg = []
wasslists_perturb4_base = []
wasslists_perturb4_convAE = []
wasslists_perturb4_mlpVAE = []
wasslists_perturb4_cnnVAE = []
wasslists_perturb4_contra = []
wasslists_perturb4_reg = []

wasslists_perturb3_hyb_base = []
wasslists_perturb3_hyb_reg = []
wasslists_perturb3_base = []
wasslists_perturb3_convAE = []
wasslists_perturb3_mlpVAE = []
wasslists_perturb3_cnnVAE = []
wasslists_perturb3_contra = []
wasslists_perturb3_reg = []

wasslists_perturb2_hyb_base = []
wasslists_perturb2_hyb_reg = []
wasslists_perturb2_base = []
wasslists_perturb2_convAE = []
wasslists_perturb2_mlpVAE = []
wasslists_perturb2_cnnVAE = []
wasslists_perturb2_contra = []
wasslists_perturb2_reg = []

wasslists_perturb1_hyb_base = []
wasslists_perturb1_hyb_reg = []
wasslists_perturb1_base = []
wasslists_perturb1_convAE = []
wasslists_perturb1_mlpVAE = []
wasslists_perturb1_cnnVAE = []
wasslists_perturb1_contra = []
wasslists_perturb1_reg = []

#####################################################################################################
ssimlists_unhyb_base = []
ssimlists_convAE = []
ssimlists_mlpVAE = []
ssimlists_cnnVAE = []
ssimlists_contra = []
ssimlists_unhyb_reg = []
ssimlists_hyb_base = []
ssimlists_hyb_reg = []

ssimlists_perturb4_hyb_base = []
ssimlists_perturb4_hyb_reg = []
ssimlists_perturb4_base = []
ssimlists_perturb4_convAE = []
ssimlists_perturb4_mlpVAE = []
ssimlists_perturb4_cnnVAE = []
ssimlists_perturb4_contra = []
ssimlists_perturb4_reg = []

ssimlists_perturb3_hyb_base = []
ssimlists_perturb3_hyb_reg = []
ssimlists_perturb3_base = []
ssimlists_perturb3_convAE = []
ssimlists_perturb3_mlpVAE = []
ssimlists_perturb3_cnnVAE = []
ssimlists_perturb3_contra = []
ssimlists_perturb3_reg = []

ssimlists_perturb2_hyb_base = []
ssimlists_perturb2_hyb_reg = []
ssimlists_perturb2_base = []
ssimlists_perturb2_convAE = []
ssimlists_perturb2_mlpVAE = []
ssimlists_perturb2_cnnVAE = []
ssimlists_perturb2_contra = []
ssimlists_perturb2_reg = []

ssimlists_perturb1_hyb_base = []
ssimlists_perturb1_hyb_reg = []
ssimlists_perturb1_base = []
ssimlists_perturb1_convAE = []
ssimlists_perturb1_mlpVAE = []
ssimlists_perturb1_cnnVAE = []
ssimlists_perturb1_contra = []
ssimlists_perturb1_reg = []

##############################################################################################

psnrlists_unhyb_base = []
psnrlists_convAE = []
psnrlists_mlpVAE = []
psnrlists_cnnVAE = []
psnrlists_contra = []
psnrlists_unhyb_reg = []
psnrlists_hyb_base = []
psnrlists_hyb_reg = []

psnrlists_perturb4_base = []
psnrlists_perturb4_convAE = []
psnrlists_perturb4_mlpVAE = []
psnrlists_perturb4_cnnVAE = []
psnrlists_perturb4_contra = []
psnrlists_perturb4_reg = []
psnrlists_perturb4_hyb_base = []
psnrlists_perturb4_hyb_reg = []

psnrlists_perturb3_base = []
psnrlists_perturb3_convAE = []
psnrlists_perturb3_mlpVAE = []
psnrlists_perturb3_cnnVAE = []
psnrlists_perturb3_contra = []
psnrlists_perturb3_reg = []
psnrlists_perturb3_hyb_base = []
psnrlists_perturb3_hyb_reg = []

psnrlists_perturb2_base = []
psnrlists_perturb2_convAE = []
psnrlists_perturb2_mlpVAE = []
psnrlists_perturb2_cnnVAE = []
psnrlists_perturb2_contra = []
psnrlists_perturb2_reg = []
psnrlists_perturb2_hyb_base = []
psnrlists_perturb2_hyb_reg = []

psnrlists_perturb1_base = []
psnrlists_perturb1_convAE = []
psnrlists_perturb1_mlpVAE = []
psnrlists_perturb1_cnnVAE = []
psnrlists_perturb1_contra = []
psnrlists_perturb1_reg = []
psnrlists_perturb1_hyb_base = []
psnrlists_perturb1_hyb_reg = []

for i in range(len(testImages)):

    testImage_normal = Normalize()(testImages[i])
    recon_normal_unhyb_base = Normalize()(unhyb_rec_bAE_test[i])
    recon_normal_convAE = Normalize()(convAE_rec_test[i])
    recon_normal_mlpVAE = Normalize()(mlpVAE_rec_test[i])
    recon_normal_cnnVAE = Normalize()(cnnVAE_rec_test[i])
    recon_normal_contra = Normalize()(contra_rec_test[i])
    recon_normal_unhyb_reg = Normalize()(unhyb_rec_rAE_test[i])
    recon_normal_hyb_base = Normalize()(hybrd_reconBase_test[i])
    recon_normal_hyb_reg = Normalize()(hybrd_reconReg_test[i])

    recon_normal_unhyb_base_perturb4 = Normalize()(unhyb_base_prturb4_recon[i])
    recon_normal_convAE_perturb4 = Normalize()(convAE_prturb4_recon[i])
    recon_normal_mlpVAE_perturb4 = Normalize()(mlpVAE_prturb4_recon[i])
    recon_normal_cnnVAE_perturb4 = Normalize()(cnnVAE_prturb4_recon[i])
    recon_normal_contra_perturb4 = Normalize()(contra_prturb4_recon[i])
    recon_normal_unhyb_reg_perturb4 = Normalize()(unhyb_reg_prturb4_recon[i])
    recon_normal_hyb_base_perturb4 = Normalize()(hyb_base_prturb4_recon[i])
    recon_normal_hyb_reg_perturb4 = Normalize()(hyb_reg_prturb4_recon[i])

    recon_normal_unhyb_base_perturb3 = Normalize()(unhyb_base_prturb3_recon[i])
    recon_normal_convAE_perturb3 = Normalize()(convAE_prturb3_recon[i])
    recon_normal_mlpVAE_perturb3 = Normalize()(mlpVAE_prturb3_recon[i])
    recon_normal_cnnVAE_perturb3 = Normalize()(cnnVAE_prturb3_recon[i])
    recon_normal_contra_perturb3 = Normalize()(contra_prturb3_recon[i])
    recon_normal_unhyb_reg_perturb3 = Normalize()(unhyb_reg_prturb3_recon[i])
    recon_normal_hyb_base_perturb3 = Normalize()(hyb_base_prturb3_recon[i])
    recon_normal_hyb_reg_perturb3 = Normalize()(hyb_reg_prturb3_recon[i])

    recon_normal_unhyb_base_perturb2 = Normalize()(unhyb_base_prturb2_recon[i])
    recon_normal_convAE_perturb2 = Normalize()(convAE_prturb2_recon[i])
    recon_normal_mlpVAE_perturb2 = Normalize()(mlpVAE_prturb2_recon[i])
    recon_normal_cnnVAE_perturb2 = Normalize()(cnnVAE_prturb2_recon[i])
    recon_normal_contra_perturb2 = Normalize()(contra_prturb2_recon[i])
    recon_normal_unhyb_reg_perturb2 = Normalize()(unhyb_reg_prturb2_recon[i])
    recon_normal_hyb_base_perturb2 = Normalize()(hyb_base_prturb2_recon[i])
    recon_normal_hyb_reg_perturb2 = Normalize()(hyb_reg_prturb2_recon[i])

    recon_normal_unhyb_base_perturb1 = Normalize()(unhyb_base_prturb1_recon[i])
    recon_normal_convAE_perturb1 = Normalize()(convAE_prturb1_recon[i])
    recon_normal_mlpVAE_perturb1 = Normalize()(mlpVAE_prturb1_recon[i])
    recon_normal_cnnVAE_perturb1 = Normalize()(cnnVAE_prturb1_recon[i])
    recon_normal_contra_perturb1 = Normalize()(contra_prturb1_recon[i])
    recon_normal_unhyb_reg_perturb1 = Normalize()(unhyb_reg_prturb1_recon[i])
    recon_normal_hyb_base_perturb1 = Normalize()(hyb_base_prturb1_recon[i])
    recon_normal_hyb_reg_perturb1 = Normalize()(hyb_reg_prturb1_recon[i])


    ###########################################################################################

    wasslists_unhyb_base.append(max(wass(testImage_normal[0], recon_normal_unhyb_base[0]), 0))
    wasslists_convAE.append(max(wass(testImage_normal[0], recon_normal_convAE[0]), 0))
    wasslists_mlpVAE.append(max(wass(testImage_normal[0], recon_normal_mlpVAE[0]), 0))
    wasslists_cnnVAE.append(max(wass(testImage_normal[0], recon_normal_cnnVAE[0]), 0))
    wasslists_contra.append(max(wass(testImage_normal[0], recon_normal_contra[0]), 0))
    wasslists_unhyb_reg.append(max(wass(testImage_normal[0], recon_normal_unhyb_reg[0]), 0))
    wasslists_hyb_base.append(max(wass(testImage_normal[0], recon_normal_hyb_base[0]), 0))
    wasslists_hyb_reg.append(max(wass(testImage_normal[0], recon_normal_hyb_reg[0]), 0))

    wasslists_perturb4_base.append(max(wass(testImage_normal[0], recon_normal_unhyb_base_perturb4[0]), 0))
    wasslists_perturb4_convAE.append(max(wass(testImage_normal[0], recon_normal_convAE_perturb4[0]), 0))
    wasslists_perturb4_mlpVAE.append(max(wass(testImage_normal[0], recon_normal_mlpVAE_perturb4[0]), 0))
    wasslists_perturb4_cnnVAE.append(max(wass(testImage_normal[0], recon_normal_cnnVAE_perturb4[0]), 0))
    wasslists_perturb4_contra.append(max(wass(testImage_normal[0], recon_normal_contra_perturb4[0]), 0))
    wasslists_perturb4_reg.append(max(wass(testImage_normal[0], recon_normal_unhyb_reg_perturb4[0]), 0))
    wasslists_perturb4_hyb_base.append(max(wass(testImage_normal[0], recon_normal_hyb_base_perturb4[0]), 0))
    wasslists_perturb4_hyb_reg.append(max(wass(testImage_normal[0], recon_normal_hyb_reg_perturb4[0]), 0))

    wasslists_perturb3_base.append(max(wass(testImage_normal[0], recon_normal_unhyb_base_perturb3[0]), 0))
    wasslists_perturb3_convAE.append(max(wass(testImage_normal[0], recon_normal_convAE_perturb3[0]), 0))
    wasslists_perturb3_mlpVAE.append(max(wass(testImage_normal[0], recon_normal_mlpVAE_perturb3[0]), 0))
    wasslists_perturb3_cnnVAE.append(max(wass(testImage_normal[0], recon_normal_cnnVAE_perturb3[0]), 0))
    wasslists_perturb3_contra.append(max(wass(testImage_normal[0], recon_normal_contra_perturb3[0]), 0))
    wasslists_perturb3_reg.append(max(wass(testImage_normal[0], recon_normal_unhyb_reg_perturb3[0]), 0))
    wasslists_perturb3_hyb_base.append(max(wass(testImage_normal[0], recon_normal_hyb_base_perturb3[0]), 0))
    wasslists_perturb3_hyb_reg.append(max(wass(testImage_normal[0], recon_normal_hyb_reg_perturb3[0]), 0))

    wasslists_perturb2_base.append(max(wass(testImage_normal[0], recon_normal_unhyb_base_perturb2[0]), 0))
    wasslists_perturb2_convAE.append(max(wass(testImage_normal[0], recon_normal_convAE_perturb2[0]), 0))
    wasslists_perturb2_mlpVAE.append(max(wass(testImage_normal[0], recon_normal_mlpVAE_perturb2[0]), 0))
    wasslists_perturb2_cnnVAE.append(max(wass(testImage_normal[0], recon_normal_cnnVAE_perturb2[0]), 0))
    wasslists_perturb2_contra.append(max(wass(testImage_normal[0], recon_normal_contra_perturb2[0]), 0))
    wasslists_perturb2_reg.append(max(wass(testImage_normal[0], recon_normal_unhyb_reg_perturb2[0]), 0))
    wasslists_perturb2_hyb_base.append(max(wass(testImage_normal[0], recon_normal_hyb_base_perturb2[0]), 0))
    wasslists_perturb2_hyb_reg.append(max(wass(testImage_normal[0], recon_normal_hyb_reg_perturb2[0]), 0))

    wasslists_perturb1_base.append(max(wass(testImage_normal[0], recon_normal_unhyb_base_perturb1[0]), 0))
    wasslists_perturb1_convAE.append(max(wass(testImage_normal[0], recon_normal_convAE_perturb1[0]), 0))
    wasslists_perturb1_mlpVAE.append(max(wass(testImage_normal[0], recon_normal_mlpVAE_perturb1[0]), 0))
    wasslists_perturb1_cnnVAE.append(max(wass(testImage_normal[0], recon_normal_cnnVAE_perturb1[0]), 0))
    wasslists_perturb1_contra.append(max(wass(testImage_normal[0], recon_normal_contra_perturb1[0]), 0))
    wasslists_perturb1_reg.append(max(wass(testImage_normal[0], recon_normal_unhyb_reg_perturb1[0]), 0))
    wasslists_perturb1_hyb_base.append(max(wass(testImage_normal[0], recon_normal_hyb_base_perturb1[0]), 0))
    wasslists_perturb1_hyb_reg.append(max(wass(testImage_normal[0], recon_normal_hyb_reg_perturb1[0]), 0))

    ############################################################################################

    ssimlists_unhyb_base.append(max(ssim(testImage_normal[0], recon_normal_unhyb_base[0], data_range=1.), 0))
    ssimlists_convAE.append(max(ssim(testImage_normal[0], recon_normal_convAE[0], data_range=1.), 0))
    ssimlists_mlpVAE.append(max(ssim(testImage_normal[0], recon_normal_mlpVAE[0], data_range=1.), 0))
    ssimlists_cnnVAE.append(max(ssim(testImage_normal[0], recon_normal_cnnVAE[0], data_range=1.), 0))
    ssimlists_contra.append(max(ssim(testImage_normal[0], recon_normal_contra[0], data_range=1.), 0))
    ssimlists_unhyb_reg.append(max(ssim(testImage_normal[0], recon_normal_unhyb_reg[0], data_range=1.), 0))
    ssimlists_hyb_base.append(max(ssim(testImage_normal[0], recon_normal_hyb_base[0], data_range=1.), 0))
    ssimlists_hyb_reg.append(max(ssim(testImage_normal[0], recon_normal_hyb_reg[0], data_range=1.), 0))

    ssimlists_perturb4_base.append(max(ssim(testImage_normal[0], recon_normal_unhyb_base_perturb4[0], data_range=1.), 0))
    ssimlists_perturb4_convAE.append(max(ssim(testImage_normal[0], recon_normal_convAE_perturb4[0], data_range=1.), 0))
    ssimlists_perturb4_mlpVAE.append(max(ssim(testImage_normal[0], recon_normal_mlpVAE_perturb4[0], data_range=1.), 0))
    ssimlists_perturb4_cnnVAE.append(max(ssim(testImage_normal[0], recon_normal_cnnVAE_perturb4[0], data_range=1.), 0))
    ssimlists_perturb4_contra.append(max(ssim(testImage_normal[0], recon_normal_contra_perturb4[0], data_range=1.), 0))
    ssimlists_perturb4_reg.append(max(ssim(testImage_normal[0], recon_normal_unhyb_reg_perturb4[0], data_range=1.), 0))
    ssimlists_perturb4_hyb_base.append(max(ssim(testImage_normal[0], recon_normal_hyb_base_perturb4[0], data_range=1.), 0))
    ssimlists_perturb4_hyb_reg.append(max(ssim(testImage_normal[0], recon_normal_hyb_reg_perturb4[0], data_range=1.), 0))

    ssimlists_perturb3_base.append(max(ssim(testImage_normal[0], recon_normal_unhyb_base_perturb3[0], data_range=1.), 0))
    ssimlists_perturb3_convAE.append(max(ssim(testImage_normal[0], recon_normal_convAE_perturb3[0], data_range=1.), 0))
    ssimlists_perturb3_mlpVAE.append(max(ssim(testImage_normal[0], recon_normal_mlpVAE_perturb3[0], data_range=1.), 0))
    ssimlists_perturb3_cnnVAE.append(max(ssim(testImage_normal[0], recon_normal_cnnVAE_perturb3[0], data_range=1.), 0))
    ssimlists_perturb3_contra.append(max(ssim(testImage_normal[0], recon_normal_contra_perturb3[0], data_range=1.), 0))
    ssimlists_perturb3_reg.append(max(ssim(testImage_normal[0], recon_normal_unhyb_reg_perturb3[0], data_range=1.), 0))
    ssimlists_perturb3_hyb_base.append(max(ssim(testImage_normal[0], recon_normal_hyb_base_perturb3[0], data_range=1.), 0))
    ssimlists_perturb3_hyb_reg.append(max(ssim(testImage_normal[0], recon_normal_hyb_reg_perturb3[0], data_range=1.), 0))

    ssimlists_perturb2_base.append(max(ssim(testImage_normal[0], recon_normal_unhyb_base_perturb2[0], data_range=1.), 0))
    ssimlists_perturb2_convAE.append(max(ssim(testImage_normal[0], recon_normal_convAE_perturb2[0], data_range=1.), 0))
    ssimlists_perturb2_mlpVAE.append(max(ssim(testImage_normal[0], recon_normal_mlpVAE_perturb2[0], data_range=1.), 0))
    ssimlists_perturb2_cnnVAE.append(max(ssim(testImage_normal[0], recon_normal_cnnVAE_perturb2[0], data_range=1.), 0))
    ssimlists_perturb2_contra.append(max(ssim(testImage_normal[0], recon_normal_contra_perturb2[0], data_range=1.), 0))
    ssimlists_perturb2_reg.append(max(ssim(testImage_normal[0], recon_normal_unhyb_reg_perturb2[0], data_range=1.), 0))
    ssimlists_perturb2_hyb_base.append(max(ssim(testImage_normal[0], recon_normal_hyb_base_perturb2[0], data_range=1.), 0))
    ssimlists_perturb2_hyb_reg.append(max(ssim(testImage_normal[0], recon_normal_hyb_reg_perturb2[0], data_range=1.), 0))

    ssimlists_perturb1_base.append(max(ssim(testImage_normal[0], recon_normal_unhyb_base_perturb1[0], data_range=1.), 0))
    ssimlists_perturb1_convAE.append(max(ssim(testImage_normal[0], recon_normal_convAE_perturb1[0], data_range=1.), 0))
    ssimlists_perturb1_mlpVAE.append(max(ssim(testImage_normal[0], recon_normal_mlpVAE_perturb1[0], data_range=1.), 0))
    ssimlists_perturb1_cnnVAE.append(max(ssim(testImage_normal[0], recon_normal_cnnVAE_perturb1[0], data_range=1.), 0))
    ssimlists_perturb1_contra.append(max(ssim(testImage_normal[0], recon_normal_contra_perturb1[0], data_range=1.), 0))
    ssimlists_perturb1_reg.append(max(ssim(testImage_normal[0], recon_normal_unhyb_reg_perturb1[0], data_range=1.), 0))
    ssimlists_perturb1_hyb_base.append(max(ssim(testImage_normal[0], recon_normal_hyb_base_perturb1[0], data_range=1.), 0))
    ssimlists_perturb1_hyb_reg.append(max(ssim(testImage_normal[0], recon_normal_hyb_reg_perturb1[0], data_range=1.), 0))

    #############################################################################################

    psnrlists_unhyb_base.append(psnr(testImage_normal[0], recon_normal_unhyb_base[0], data_range=1.))
    psnrlists_convAE.append(psnr(testImage_normal[0], recon_normal_convAE[0], data_range=1.))
    psnrlists_mlpVAE.append(psnr(testImage_normal[0], recon_normal_mlpVAE[0], data_range=1.))
    psnrlists_cnnVAE.append(psnr(testImage_normal[0], recon_normal_cnnVAE[0], data_range=1.))
    psnrlists_contra.append(psnr(testImage_normal[0], recon_normal_contra[0], data_range=1.))
    psnrlists_unhyb_reg.append(psnr(testImage_normal[0], recon_normal_unhyb_reg[0], data_range=1.))
    psnrlists_hyb_base.append(psnr(testImage_normal[0], recon_normal_hyb_base[0], data_range=1.))
    psnrlists_hyb_reg.append(psnr(testImage_normal[0], recon_normal_hyb_reg[0], data_range=1.))

    psnrlists_perturb4_base.append(psnr(testImage_normal[0], recon_normal_unhyb_base_perturb4[0], data_range=1.))
    psnrlists_perturb4_convAE.append(psnr(testImage_normal[0], recon_normal_convAE_perturb4[0], data_range=1.))
    psnrlists_perturb4_mlpVAE.append(psnr(testImage_normal[0], recon_normal_mlpVAE_perturb4[0], data_range=1.))
    psnrlists_perturb4_cnnVAE.append(psnr(testImage_normal[0], recon_normal_cnnVAE_perturb4[0], data_range=1.))
    psnrlists_perturb4_contra.append(psnr(testImage_normal[0], recon_normal_contra_perturb4[0], data_range=1.))
    psnrlists_perturb4_reg.append(psnr(testImage_normal[0], recon_normal_unhyb_reg_perturb4[0], data_range=1.))
    psnrlists_perturb4_hyb_base.append(psnr(testImage_normal[0], recon_normal_hyb_base_perturb4[0], data_range=1.))
    psnrlists_perturb4_hyb_reg.append(psnr(testImage_normal[0], recon_normal_hyb_reg_perturb4[0], data_range=1.))

    psnrlists_perturb3_base.append(psnr(testImage_normal[0], recon_normal_unhyb_base_perturb3[0], data_range=1.))
    psnrlists_perturb3_convAE.append(psnr(testImage_normal[0], recon_normal_convAE_perturb3[0], data_range=1.))
    psnrlists_perturb3_mlpVAE.append(psnr(testImage_normal[0], recon_normal_mlpVAE_perturb3[0], data_range=1.))
    psnrlists_perturb3_cnnVAE.append(psnr(testImage_normal[0], recon_normal_cnnVAE_perturb3[0], data_range=1.))
    psnrlists_perturb3_contra.append(psnr(testImage_normal[0], recon_normal_contra_perturb3[0], data_range=1.))
    psnrlists_perturb3_reg.append(psnr(testImage_normal[0], recon_normal_unhyb_reg_perturb3[0], data_range=1.))
    psnrlists_perturb3_hyb_base.append(psnr(testImage_normal[0], recon_normal_hyb_base_perturb3[0], data_range=1.))
    psnrlists_perturb3_hyb_reg.append(psnr(testImage_normal[0], recon_normal_hyb_reg_perturb3[0], data_range=1.))

    psnrlists_perturb2_base.append(psnr(testImage_normal[0], recon_normal_unhyb_base_perturb2[0], data_range=1.))
    psnrlists_perturb2_convAE.append(psnr(testImage_normal[0], recon_normal_convAE_perturb2[0], data_range=1.))
    psnrlists_perturb2_mlpVAE.append(psnr(testImage_normal[0], recon_normal_mlpVAE_perturb2[0], data_range=1.))
    psnrlists_perturb2_cnnVAE.append(psnr(testImage_normal[0], recon_normal_cnnVAE_perturb2[0], data_range=1.))
    psnrlists_perturb2_contra.append(psnr(testImage_normal[0], recon_normal_contra_perturb2[0], data_range=1.))
    psnrlists_perturb2_reg.append(psnr(testImage_normal[0], recon_normal_unhyb_reg_perturb2[0], data_range=1.))
    psnrlists_perturb2_hyb_base.append(psnr(testImage_normal[0], recon_normal_hyb_base_perturb2[0], data_range=1.))
    psnrlists_perturb2_hyb_reg.append(psnr(testImage_normal[0], recon_normal_hyb_reg_perturb2[0], data_range=1.))

    psnrlists_perturb1_base.append(psnr(testImage_normal[0], recon_normal_unhyb_base_perturb1[0], data_range=1.))
    psnrlists_perturb1_convAE.append(psnr(testImage_normal[0], recon_normal_convAE_perturb1[0], data_range=1.))
    psnrlists_perturb1_mlpVAE.append(psnr(testImage_normal[0], recon_normal_mlpVAE_perturb1[0], data_range=1.))
    psnrlists_perturb1_cnnVAE.append(psnr(testImage_normal[0], recon_normal_cnnVAE_perturb1[0], data_range=1.))
    psnrlists_perturb1_contra.append(psnr(testImage_normal[0], recon_normal_contra_perturb1[0], data_range=1.))
    psnrlists_perturb1_reg.append(psnr(testImage_normal[0], recon_normal_unhyb_reg_perturb1[0], data_range=1.))
    psnrlists_perturb1_hyb_base.append(psnr(testImage_normal[0], recon_normal_hyb_base_perturb1[0], data_range=1.))
    psnrlists_perturb1_hyb_reg.append(psnr(testImage_normal[0], recon_normal_hyb_reg_perturb1[0], data_range=1.))


# reconstruction loss after training the model completely
loss_tre = torch.mean(((unhyb_base_prturb4_recon - testImages)**2)*0.5)


# reconstruction loss after training the model completely
loss_tre = torch.mean(((unhyb_reg_prturb4_recon - testImages)**2)*0.5)


import matplotlib.pyplot as plt
import numpy as np

print('Wasserstein Error of reconstruction on test data')
#fracs = ['MLP-AE', 'convAE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#wasss = np.load('/path/to/folder/Wasserstein Error/Wasserstein Error_data_base_test.npy')
#wasss = [wasslists_unhyb_base, wasslists_unhyb_base,  wasslists_unhyb_reg, wasslists_hyb_base, wasslists_hyb_reg]
wasss = [wasslists_unhyb_base, wasslists_convAE, wasslists_mlpVAE, wasslists_cnnVAE, wasslists_contra, wasslists_unhyb_reg, wasslists_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('Wasserstein Error of reconstruction on test data')
ax1.boxplot(list(wasss))
#ax1.set_xlabel('Models', fontsize=16)
ax1.set_ylabel('Wasserstein Error', fontsize=10)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/Wasserstein Error_directReconOfTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()



print('SSIM of reconstruction on test data')
#fracs = ['MLP-AE', 'convAE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
#ssims = [ssimlists_unhyb_base, ssimlists_unhyb_base,  ssimlists_unhyb_reg, ssimlists_hyb_base, ssimlists_hyb_reg]
ssims = [ssimlists_unhyb_base, ssimlists_convAE, ssimlists_mlpVAE, ssimlists_cnnVAE, ssimlists_contra, ssimlists_unhyb_reg, ssimlists_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('SSIM of reconstruction on test data')
ax1.boxplot(list(ssims))
#ax1.set_xlabel('Models', fontsize=16)
ax1.set_ylabel('SSIM', fontsize=10)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/SSIM_directReconOfTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()



#import matplotlib.pyplot as plt
#import numpy as np
print('PSNR of reconstruction on test data')
#fracs = ['MLP-AE','AE-REG', 'Hybrid\nMLP-AE', 'Hybrid\nAE-REG']
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
#psnrs = [psnrlists_unhyb_base, psnrlists_convAE, psnrlists_unhyb_reg, psnrlists_hyb_base, psnrlists_hyb_reg]
psnrs = [psnrlists_unhyb_base, psnrlists_convAE, psnrlists_mlpVAE, psnrlists_cnnVAE, psnrlists_contra, psnrlists_unhyb_reg, psnrlists_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('PSNR of reconstruction on test data')
ax1.boxplot(list(psnrs))
#ax1.set_xlabel('Models', fontsize=16)
ax1.set_ylabel('PSNR(dB)', fontsize=10)
ax1.set_ylim([0,28])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/PSNR_directReconOfTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()


#Perturbation test for noise

print('PSNR of reconstruction of 70 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
psnrs = [psnrlists_perturb4_base, psnrlists_perturb4_convAE, psnrlists_perturb4_mlpVAE, psnrlists_perturb4_cnnVAE, psnrlists_perturb4_contra, psnrlists_perturb4_reg, psnrlists_perturb4_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('PSNR of reconstruction of 70 % noised  test data')
ax1.boxplot(list(psnrs))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('PSNR(dB)', fontsize=10)
ax1.set_ylim([0,25])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/PSNR_ReconOf70percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(psnrlists_perturb4_base))
print(np.mean(psnrlists_perturb4_convAE))
print(np.mean(psnrlists_perturb4_mlpVAE))
print(np.mean(psnrlists_perturb4_cnnVAE))
print(np.mean(psnrlists_perturb4_contra))
print(np.mean(psnrlists_perturb4_reg))
print(np.mean(psnrlists_perturb4_hyb_reg))


#Perturbation test for noise
print('PSNR of reconstruction of 50 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#fracs = ['MLP-AE','AE-REG', 'Hybrid MLP-AE', 'Hybrid AE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
psnrs = [psnrlists_perturb3_base, psnrlists_perturb3_convAE, psnrlists_perturb3_mlpVAE, psnrlists_perturb3_cnnVAE, psnrlists_perturb3_contra, psnrlists_perturb3_reg, psnrlists_perturb3_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('PSNR of reconstruction of 50 % noised  test data')
ax1.boxplot(list(psnrs))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('PSNR(dB)', fontsize=10)
ax1.set_ylim([0,25])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/PSNR_ReconOf50percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(psnrlists_perturb3_base))
print(np.mean(psnrlists_perturb3_convAE))
print(np.mean(psnrlists_perturb3_mlpVAE))
print(np.mean(psnrlists_perturb3_cnnVAE))
print(np.mean(psnrlists_perturb3_contra))
print(np.mean(psnrlists_perturb3_reg))
print(np.mean(psnrlists_perturb3_hyb_reg))


#Perturbation test for noise
print('PSNR of reconstruction of 20 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
psnrs = [psnrlists_perturb2_base, psnrlists_perturb2_convAE, psnrlists_perturb2_mlpVAE, psnrlists_perturb2_cnnVAE, psnrlists_perturb2_contra, psnrlists_perturb2_reg, psnrlists_perturb2_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('PSNR of reconstruction of 20 % noised  test data')
ax1.boxplot(list(psnrs))
#ax1.set_xlabel('Models', fontsize=16)
ax1.set_ylabel('PSNR(dB)', fontsize=10)
ax1.set_ylim([0,25])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/PSNR_ReconOf20percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(psnrlists_perturb2_base))
print(np.mean(psnrlists_perturb2_convAE))
print(np.mean(psnrlists_perturb2_mlpVAE))
print(np.mean(psnrlists_perturb2_cnnVAE))
print(np.mean(psnrlists_perturb2_contra))
print(np.mean(psnrlists_perturb2_reg))
print(np.mean(psnrlists_perturb2_hyb_reg))



#Perturbation test for noise
print('PSNR of reconstruction of 10 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
psnrs = [psnrlists_perturb1_base, psnrlists_perturb1_convAE, psnrlists_perturb1_mlpVAE, psnrlists_perturb1_cnnVAE, psnrlists_perturb1_contra, psnrlists_perturb1_reg, psnrlists_perturb1_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('PSNR of reconstruction of 10 % noised  test data')
ax1.boxplot(list(psnrs))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('PSNR(dB)', fontsize=10)
ax1.set_ylim([0,25])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/PSNR_ReconOf10percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(psnrlists_perturb1_base))
print(np.mean(psnrlists_perturb1_convAE))
print(np.mean(psnrlists_perturb1_mlpVAE))
print(np.mean(psnrlists_perturb1_cnnVAE))
print(np.mean(psnrlists_perturb1_contra))
print(np.mean(psnrlists_perturb1_reg))
print(np.mean(psnrlists_perturb1_hyb_reg))


#Perturbation test for noise
print('SSIM of reconstruction of 70 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
ssims = [ssimlists_perturb4_base, ssimlists_perturb4_convAE, ssimlists_perturb4_mlpVAE, ssimlists_perturb4_cnnVAE, ssimlists_perturb4_contra, ssimlists_perturb4_reg, ssimlists_perturb4_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('SSIM of reconstruction of 70 % noised  test data')
ax1.boxplot(list(ssims))
#ax1.set_xlabel('Models', fontsize=16)
ax1.set_ylabel('SSIM', fontsize=10)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/SSIM_ReconOf70percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(ssimlists_perturb4_base))
print(np.mean(ssimlists_perturb4_convAE))
print(np.mean(ssimlists_perturb4_mlpVAE))
print(np.mean(ssimlists_perturb4_cnnVAE))
print(np.mean(ssimlists_perturb4_contra))
print(np.mean(ssimlists_perturb4_reg))
print(np.mean(ssimlists_perturb4_hyb_reg))


#Perturbation test for noise

print('SSIM of reconstruction of 50 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
ssims = [ssimlists_perturb3_base, ssimlists_perturb3_convAE, ssimlists_perturb3_mlpVAE, ssimlists_perturb3_cnnVAE, ssimlists_perturb3_contra, ssimlists_perturb3_reg, ssimlists_perturb3_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('SSIM of reconstruction of 50 % noised  test data')
ax1.boxplot(list(ssims))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('SSIM', fontsize=10)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/SSIM_ReconOf50percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(ssimlists_perturb3_base))
print(np.mean(ssimlists_perturb3_convAE))
print(np.mean(ssimlists_perturb3_mlpVAE))
print(np.mean(ssimlists_perturb3_cnnVAE))
print(np.mean(ssimlists_perturb3_contra))
print(np.mean(ssimlists_perturb3_reg))
print(np.mean(ssimlists_perturb3_hyb_reg))

#Perturbation test for noise
print('SSIM of reconstruction of 20 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
ssims = [ssimlists_perturb2_base, ssimlists_perturb2_convAE, ssimlists_perturb2_mlpVAE, ssimlists_perturb2_cnnVAE, ssimlists_perturb2_contra, ssimlists_perturb2_reg, ssimlists_perturb2_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('SSIM of reconstruction of 20 % noised  test data')
ax1.boxplot(list(ssims))
#ax1.set_xlabel('Models', fontsize=15)
ax1.set_ylabel('SSIM', fontsize=10)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/SSIM_ReconOf20percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(ssimlists_perturb2_base))
print(np.mean(ssimlists_perturb2_convAE))
print(np.mean(ssimlists_perturb2_mlpVAE))
print(np.mean(ssimlists_perturb2_cnnVAE))
print(np.mean(ssimlists_perturb2_contra))
print(np.mean(ssimlists_perturb2_reg))
print(np.mean(ssimlists_perturb2_hyb_reg))


#Perturbation test for noise
print('SSIM of reconstruction of 10 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#ssims = np.load('/path/to/folder/SSIM/SSIM_data_base_test.npy')
ssims = [ssimlists_perturb1_base, ssimlists_perturb1_convAE, ssimlists_perturb1_mlpVAE, ssimlists_perturb1_cnnVAE, ssimlists_perturb1_contra, ssimlists_perturb1_reg, ssimlists_perturb1_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('SSIM of reconstruction of 10 % noised  test data')
ax1.boxplot(list(ssims))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('SSIM', fontsize=10)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/SSIM_ReconOf10percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()


print(np.mean(ssimlists_perturb1_base))
print(np.mean(ssimlists_perturb1_convAE))
print(np.mean(ssimlists_perturb1_mlpVAE))
print(np.mean(ssimlists_perturb1_cnnVAE))
print(np.mean(ssimlists_perturb1_contra))
print(np.mean(ssimlists_perturb1_reg))
print(np.mean(ssimlists_perturb1_hyb_reg))


#Perturbation test for noise
print('Wasserstein Error of reconstruction of 70 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#wasss = np.load('/path/to/folder/Wasserstein Error/Wasserstein Error_data_base_test.npy')
wasss = [wasslists_perturb4_base, wasslists_perturb4_convAE, wasslists_perturb4_mlpVAE, wasslists_perturb4_cnnVAE, wasslists_perturb4_contra, wasslists_perturb4_reg, wasslists_perturb4_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('Wasserstein Error of reconstruction of 70 % noised  test data')
ax1.boxplot(list(wasss))
#ax1.set_xlabel('Models', fontsize=16)
ax1.set_ylabel('Wasserstein Error', fontsize=10)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/Wasserstein_Error_ReconOf70percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(wasslists_perturb4_base))
print(np.mean(wasslists_perturb4_convAE))
print(np.mean(wasslists_perturb4_mlpVAE))
print(np.mean(wasslists_perturb4_cnnVAE))
print(np.mean(wasslists_perturb4_contra))
print(np.mean(wasslists_perturb4_reg))
print(np.mean(wasslists_perturb4_hyb_reg))


#Perturbation test for noise

print('Wasserstein Error of reconstruction of 50 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#wasss = np.load('/path/to/folder/Wasserstein Error/Wasserstein Error_data_base_test.npy')
wasss = [wasslists_perturb3_base, wasslists_perturb3_convAE, wasslists_perturb3_mlpVAE, wasslists_perturb3_cnnVAE, wasslists_perturb3_contra, wasslists_perturb3_reg, wasslists_perturb3_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('Wasserstein Error of reconstruction of 50 % noised  test data')
ax1.boxplot(list(wasss))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('Wasserstein Error', fontsize=10)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/Wasserstein_Error_ReconOf50percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(wasslists_perturb3_base))
print(np.mean(wasslists_perturb3_convAE))
print(np.mean(wasslists_perturb3_mlpVAE))
print(np.mean(wasslists_perturb3_cnnVAE))
print(np.mean(wasslists_perturb3_contra))
print(np.mean(wasslists_perturb3_reg))
print(np.mean(wasslists_perturb3_hyb_reg))

#Perturbation test for noise
print('Wasserstein Error of reconstruction of 20 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#wasss = np.load('/path/to/folder/Wasserstein Error/Wasserstein Error_data_base_test.npy')
wasss = [wasslists_perturb2_base, wasslists_perturb2_convAE, wasslists_perturb2_mlpVAE, wasslists_perturb2_cnnVAE, wasslists_perturb2_contra, wasslists_perturb2_reg, wasslists_perturb2_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('Wasserstein Error of reconstruction of 20 % noised  test data')
ax1.boxplot(list(wasss))
#ax1.set_xlabel('Models', fontsize=15)
ax1.set_ylabel('Wasserstein Error', fontsize=10)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/Wasserstein_Error_ReconOf20percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()

print(np.mean(wasslists_perturb2_base))
print(np.mean(wasslists_perturb2_convAE))
print(np.mean(wasslists_perturb2_mlpVAE))
print(np.mean(wasslists_perturb2_cnnVAE))
print(np.mean(wasslists_perturb2_contra))
print(np.mean(wasslists_perturb2_reg))
print(np.mean(wasslists_perturb2_hyb_reg))


#Perturbation test for noise
print('Wasserstein Error of reconstruction of 10 % noised  test data')
fracs = ['MLP-AE', 'CNN-AE', 'MLP-VAE', 'CNN-VAE', 'ContraAE', 'AE-REG', 'Hybrid\nAE-REG']
#wasss = np.load('/path/to/folder/Wasserstein Error/Wasserstein Error_data_base_test.npy')
wasss = [wasslists_perturb1_base, wasslists_perturb1_convAE, wasslists_perturb1_mlpVAE, wasslists_perturb1_cnnVAE, wasslists_perturb1_contra, wasslists_perturb1_reg, wasslists_perturb1_hyb_reg]
fig1, ax1 = plt.subplots()
#ax1.set_title('Wasserstein Error of reconstruction of 10 % noised  test data')
ax1.boxplot(list(wasss))
#ax1.set_xlabel('Models', fontsize=10)
ax1.set_ylabel('Wasserstein Error', fontsize=10)
ax1.set_ylim([0,1])
plt.xticks([1, 2, 3, 4, 5, 6, 7], [str(s) for s in fracs], fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./all_results/MRI_box_plots/Wasserstein_Error_ReconOf10percentNoiseedTestData_LossBal'+str(alpha)+'Lat_dim'+str(latent_dim)+'TDA_'+str(frac)+'LSTSQ_deg_'+str(deg_quad)+'.png')
plt.show()


print(np.mean(wasslists_perturb1_base))
print(np.mean(wasslists_perturb1_convAE))
print(np.mean(wasslists_perturb1_mlpVAE))
print(np.mean(wasslists_perturb1_cnnVAE))
print(np.mean(wasslists_perturb1_contra))
print(np.mean(wasslists_perturb1_reg))
print(np.mean(wasslists_perturb1_hyb_reg))