import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

import os

import sys
sys.path.append('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders')

import re
#from datasets import getMNIST, getFashionMNIST, getCifar10, getDataset
import copy

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')
from models import AE
#from vae import BetaVAE
from activations import Sin
from regularisers import computeC1Loss
#from models_circle import MLPVAE

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

filename = "table_model_reg_legendre_legendre.txt"


loadableDatas = ["train", "test"]
choosenData = loadableDatas[1]
availableModels = ["baseline", "regularized"]
modelSelected = availableModels[1]

coeff_sol_method = "LSTQS"

Hybrid_poly_deg = 20



ChoosenImageIndex = 9

hiddens = [100]
hidden_size = 100

alphas = [0.1]
alpha = 0.5

latent_dims = [2,3,4,5,6,7,8,9,10]
latent_dim = 6

fracs = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
frac = 0.4

prozs = [0.1, 0.2, 0.5, 0.7, ]

rand_perturb = []
orig_perturb = []
rec_perturb = []
#model = AE([1,32,32], hidden_size, latent_dim, 3, Sin()).to('cuda')
path_ = '/home/ramana44/autoencoder_regulrization_conf_tasks/models/'
paths = [path_+'model_base_legendre_', path_+'model_reg_trainingData_', path_+'model_reg_legendre_', '/home/willma32/regularizedautoencoder/output/FMNIST_vae/model_reg_']

names = ['baseline', 'contractive', 'legendre', 'vae']

#cj = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
flh = transforms.RandomHorizontalFlip(p=1.)
flv = transforms.RandomVerticalFlip(p=1.)
#cr = transforms.RandomResizedCrop(size=(32, 32))

labels = ['baseline', 'contractive', 'legendre', 'vae']
path_file = '/home/ramana44/autoencoder_regulrization_conf_tasks/FMNIST_samples/'
path_to_dir = '/home/ramana44/autoencoder_regulrization_conf_tasks_hybrid/output_all_lstqs/'

#hidden, alpha, latent, frac, name, psnr_orig, ssim_orig, psnr_cj, ssim_cj, psnr_flh, ssim_flh, psnr_flv, ssim_flhv, psnr_cr, ssim_cr, psnr_rot, ssim_rot, psnr_n1, ssim_n1, ... psnr_n5, ssim_n5
#all_results = np.zeros((len(hiddens)*len(alphas)*len(latent_dims)*len(fracs)*len(names), 49))
global_ind = 0

rand_perturb = []
orig_perturb = []
rec_perturb = []
path_to_model = paths[2] + str(alpha)+'_'+str(latent_dim)+'_'+str(hidden_size)+'_'+str(frac)

deg_quad = Hybrid_poly_deg


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

test_data = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images.pt')

print('orig.shape', test_data.shape)

Images_pert_10_percent = torch.tensor([])
Images_pert_20_percent = torch.tensor([])
Images_pert_50_percent = torch.tensor([])
Images_pert_70_percent = torch.tensor([])
for ChoosenImageIndex in range(200):

    orig = np.array(test_data[ChoosenImageIndex])
    #print(orig.shape)


    orig = orig.reshape(1,1024)

    rand_perturb = []
    for proz in prozs:
        rand_perturb.append(np.random.rand(1, 1024)*(np.max(orig)-np.min(orig))*proz)

    orig_perturb = []
    for rand_transform in rand_perturb:
        orig_perturb.append(torch.from_numpy(np.add(orig,rand_transform)).reshape(1,32,32).to(device))


    '''orig_perturbc0 = get_all_thetas(orig_perturb[0].reshape(1,1024))
    Images_pert_10_percent = torch.cat((Images_pert_10_percent,orig_perturbc0.unsqueeze(0)))
    orig_perturbc1 = get_all_thetas(orig_perturb[1].reshape(1,1024))
    Images_pert_20_percent = torch.cat((Images_pert_20_percent,orig_perturbc1.unsqueeze(0)))
    orig_perturbc2 = get_all_thetas(orig_perturb[2].reshape(1,1024))
    Images_pert_50_percent = torch.cat((Images_pert_50_percent,orig_perturbc2.unsqueeze(0)))
    orig_perturbc3 = get_all_thetas(orig_perturb[3].reshape(1,1024))
    Images_pert_70_percent = torch.cat((Images_pert_70_percent,orig_perturbc3.unsqueeze(0)))'''


    #orig_perturbc0 = get_all_thetas(orig_perturb[0].reshape(1,1024))
    Images_pert_10_percent = torch.cat((Images_pert_10_percent,orig_perturb[0].unsqueeze(0)))
    #orig_perturbc1 = get_all_thetas(orig_perturb[1].reshape(1,1024))
    Images_pert_20_percent = torch.cat((Images_pert_20_percent,orig_perturb[1].unsqueeze(0)))
    #orig_perturbc2 = get_all_thetas(orig_perturb[2].reshape(1,1024))
    Images_pert_50_percent = torch.cat((Images_pert_50_percent, orig_perturb[2].unsqueeze(0)))
    #orig_perturbc3 = get_all_thetas(orig_perturb[3].reshape(1,1024))
    Images_pert_70_percent = torch.cat((Images_pert_70_percent,orig_perturb[3].unsqueeze(0)))

    #print('len(orig_perturb)', len(orig_perturb))
#Images_pert_10_percent = Images_pert_10_percent.unsqueeze(1)
#Images_pert_20_percent = Images_pert_20_percent.unsqueeze(1)
#Images_pert_50_percent = Images_pert_50_percent.unsqueeze(1)
#Images_pert_70_percent = Images_pert_70_percent.unsqueeze(1)

print('Images_pert_10_percent.shape', Images_pert_10_percent.shape)
print('Images_pert_20_percent.shape', Images_pert_20_percent.shape)
print('Images_pert_50_percent.shape', Images_pert_50_percent.shape)
print('Images_pert_70_percent.shape', Images_pert_70_percent.shape)

'''torch.save(Images_pert_10_percent, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/coeffs_saved/LSTSQ_N10_testcoeffs_FMNIST_dq'+str(deg_quad)+'.pt')
torch.save(Images_pert_20_percent, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/coeffs_saved/LSTSQ_N20_testcoeffs_FMNIST_dq'+str(deg_quad)+'.pt')
torch.save(Images_pert_50_percent, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/coeffs_saved/LSTSQ_N50_testcoeffs_FMNIST_dq'+str(deg_quad)+'.pt')
torch.save(Images_pert_70_percent, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/coeffs_saved/LSTSQ_N70_testcoeffs_FMNIST_dq'+str(deg_quad)+'.pt')'''


torch.save(Images_pert_10_percent, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images_N_10.pt')
torch.save(Images_pert_20_percent, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images_N_20.pt')
torch.save(Images_pert_50_percent, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images_N_50.pt')
torch.save(Images_pert_70_percent, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/'+choosenData+'Images_N_70.pt')

#testCoeffs = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/coeffs_saved/LSTSQ_testcoeffs_FMNIST_dq'+str(deg_quad)+'.pt',map_location=torch.device('cpu'))


#print('testCoeffs.shape', testCoeffs.shape)








#testRK = get_all_thetas(orig)

#print('testRK', testRK.shape)
