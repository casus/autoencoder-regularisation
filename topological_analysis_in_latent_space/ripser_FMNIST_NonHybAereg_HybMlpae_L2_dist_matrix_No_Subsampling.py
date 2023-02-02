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
from matplotlib.pyplot import figure

Analys_size = 200


from models_un import AE_un
from models import AE
from activations import Sin


# load trained rAE and bAE
latent_dims = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
all_hyb_base_models = []
all_hyb_reg_models = []
all_test_coeffs = []
all_X_p = []
for lat_dim in latent_dims:
    deg_quad = 20
    u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
    x = np.linspace(-1,1,32)
    X_p = u_ob.data_axes([x,x]).T

    testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/testImages.pt',map_location=torch.device('cpu'))
    testCoeffs = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/coeffs_saved/LSTSQ_testcoeffs_FMNIST_dq'+str(deg_quad)+'.pt',map_location=torch.device('cpu'))

    testImages = testImages[:Analys_size]
    testCoeffs = testCoeffs[:Analys_size]

    path_hyb = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_space/output/MRT_full/test_run_saving/'
    path_unhyb = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'

    #specify hyperparameters
    reg_nodes_sampling = 'legendre'
    alpha = 0.5
    frac = 0.4
    hidden_size = 100
    deg_poly = 21
    deg_poly_forRK = 21
    latent_dim = lat_dim
    lr = 0.0001
    no_layers = 3
    no_epochs= 100
    name_hyb = '_'+reg_nodes_sampling+'_'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly_forRK)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)
    name_unhyb = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

    inp_dim_hyb = (deg_quad+1)*(deg_quad+1)

    inp_dim_unhyb = [1,32,32]

    model_reg = AE_un(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
    model_base = AE_un(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

    RK_model_reg = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
    RK_model_base = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

    RK_model_reg.load_state_dict(torch.load(path_hyb+'model_regLSTQS'+str(deg_quad)+''+name_hyb, map_location=torch.device('cpu')))
    RK_model_base.load_state_dict(torch.load(path_hyb+'model_baseLSTQS'+str(deg_quad)+''+name_hyb, map_location=torch.device('cpu')))

    model_reg.load_state_dict(torch.load(path_unhyb+'model_reg_TDA'+name_unhyb, map_location=torch.device('cpu')))
    #model_base.load_state_dict(torch.load(path_unhyb+'model_base_TDA'+name_unhyb, map_location=torch.device('cpu')))

    all_hyb_base_models.append(RK_model_base)
    all_hyb_reg_models.append(RK_model_reg)
    all_test_coeffs.append(testCoeffs)
    all_X_p.append(X_p)



all_rec_rAE_test = []
all_rec_bAE_test = []
for i in range(len(latent_dims)):
    rec_rAE_test = all_hyb_reg_models[i].encoder(all_test_coeffs[i].float())#.view(all_test_coeffs[i].shape)
    rec_bAE_test = all_hyb_base_models[i].encoder(all_test_coeffs[i].float())#.view(all_test_coeffs[i].shape)
    
    rec_rAE_test = torch.tensor(rec_rAE_test, requires_grad=False)
    rec_bAE_test = torch.tensor(rec_bAE_test, requires_grad=False)

    all_rec_rAE_test.append(rec_rAE_test)
    all_rec_bAE_test.append(rec_bAE_test)


#plt.scatter(all_rec_bAE_test[4][:,0], all_rec_bAE_test[4][:,1])
#plt.close()
def _compute_distance_matrix(x, p=2):
    x_flat = x.view(x.size(0), -1)

    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)

    return distances

#dist_matrix_Lat_10 = _compute_distance_matrix(all_rec_rAE_test[4], p=2)


#from ripser import ripser
import ripser
import persim
from persim import plot_diagrams


#Persistent Homology of first 200 test images using L2 distance matrix
dist_matrix_Lat_ori = _compute_distance_matrix(testImages, p=2)
diagrams_ori = ripser.ripser(dist_matrix_Lat_ori.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
plot_diagrams(diagrams_ori, show=True)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('Birth',fontsize=15)
plt.ylabel('Death' ,fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/topological_analysis_in_latent_space/plots_from_topology/PH_FMNIST_testImages_L2_dist_matrix_NoSubsampling_size200.png')
plt.close()

for i in range(len(latent_dims)):
    #print(latent_dims[i])

    #Persistent Homology of first 200 test images in all latent dimensions using L2 distance matrix HybAEREG
    dist_matrix_Lat = _compute_distance_matrix(all_rec_rAE_test[i], p=2)
    diagrams = ripser.ripser(dist_matrix_Lat.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
    plot_diagrams(diagrams, show=True)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Birth',fontsize=15)
    plt.ylabel('Death' ,fontsize=15)
    plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/topological_analysis_in_latent_space/plots_from_topology/PH_FMNIST_NonHybAeRegLatDim'+str(latent_dims[i])+'_L2_dist_matrix_NoSubsampling_size200.png')
    plt.close()


# L2 distances of Persistent homology signatures in latent spaces from the persistent homology signatures of original point cloud of images

all_mse = []
for i in range(len(latent_dims)):
    dist_matrix_Lat = _compute_distance_matrix(all_rec_rAE_test[i], p=2)
    diagrams = ripser.ripser(dist_matrix_Lat.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
    mse = np.mean(np.sqrt((diagrams[0][:-1] - diagrams_ori[0][:-1])**2))
    all_mse.append(mse)
    pre_diag = diagrams
figure(figsize=(8, 6), dpi=100)
plt.plot(latent_dims, all_mse)
plt.xticks(latent_dims, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('Latent Dimension',fontsize=15)
plt.ylabel('L-2 Distance' ,fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/topological_analysis_in_latent_space/plots_from_topology/AllLatDimPH_distances_from_originalImageCloudPH_FMNIST_NonHybAeReg_L2_dist_matrix_NoSubsampling_size200.png')
plt.close()



for i in range(len(latent_dims)):
    #print(latent_dims[i])
    #Persistent Homology of first 200 test images in all latent dimensions using L2 distance matrix HyMLPAE
    dist_matrix_Lat = _compute_distance_matrix(all_rec_bAE_test[i], p=2)
    diagrams = ripser.ripser(dist_matrix_Lat.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
    plot_diagrams(diagrams, show=True)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Birth',fontsize=15)
    plt.ylabel('Death' ,fontsize=15)
    plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/topological_analysis_in_latent_space/plots_from_topology/PH_FMNIST_NonHybMlpaeLatDim'+str(latent_dims[i])+'_L2_dist_matrix_NoSubsampling_size200.png')
    plt.close()


all_mse = []
for i in range(len(latent_dims)):
    dist_matrix_Lat = _compute_distance_matrix(all_rec_bAE_test[i], p=2)
    diagrams = ripser.ripser(dist_matrix_Lat.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
    mse = np.mean(np.sqrt((diagrams[0][:-1] - diagrams_ori[0][:-1])**2))
    all_mse.append(mse)
    pre_diag = diagrams
figure(figsize=(8, 6), dpi=100)
plt.plot(latent_dims, all_mse)
plt.xticks(latent_dims, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('Latent Dimension',fontsize=15)
plt.ylabel('L-2 Distance' ,fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/topological_analysis_in_latent_space/plots_from_topology/AllLatDimPH_distances_from_originalImageCloudPH_FMNIST_NonHybMlpae_L2_dist_matrix_NoSubsampling_size200.png')
plt.close()