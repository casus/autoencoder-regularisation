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


trainImages = torch.load('/home/ramana44/oasis_mri_2d_slices_hybridAlphaHalf_81coeffs_RK_LossWithImage/savedDatasetAndCoeffs/trainDataSet.pt',map_location=torch.device('cpu'))
trainCoeffs = torch.load('/home/ramana44/oasis_mri_2d_slices_hybridAlphaHalf_81coeffs_RK_LossWithImage/savedDatasetAndCoeffs/trainDataRK_coeffs.pt',map_location=torch.device('cpu'))


testImages = torch.load('/home/ramana44/oasis_mri_2d_slices_hybridAlphaHalf_81coeffs_RK_LossWithImage/savedDatasetAndCoeffs/testDataSet.pt',map_location=torch.device('cpu'))
testCoeffs = torch.load('/home/ramana44/oasis_mri_2d_slices_hybridAlphaHalf_81coeffs_RK_LossWithImage/savedDatasetAndCoeffs/testDataRK_coeffs.pt',map_location=torch.device('cpu'))

Analys_size = 200

trainImages = trainImages[:Analys_size]
trainCoeffs = trainCoeffs[:Analys_size]

testImages = testImages[:Analys_size]
testCoeffs = testCoeffs[:Analys_size]

# load trained rAE and bAE
from models import AE
from activations import Sin

# PSNR and SSIM calculations


# function of batch psnr
def batch_psnr(batch_prediction, batch_target):
    avg_psnr = 0
    for i in range(batch_prediction.size(0)):
        mse = F.mse_loss(batch_prediction[i], batch_target[i])
        if mse > 0.:
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr
        else:
            avg_psnr += 100.

    return (avg_psnr / batch_prediction.size(0)).item()

path = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/output/MRT_full/test_run_saving/'
#specify hyperparameters
reg_nodes_sampling = 'legendre'
alpha = 0.001
frac = 0.8
hidden_size = 1000
deg_poly = 20
latent_dim = 18
lr = 0.0001
no_layers = 5
#no_epochs=?
name = '_'+reg_nodes_sampling+'_'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

#no_channels, dx, dy = (train_loader_alz.dataset.__getitem__(1).shape)
#inp_dim = [no_channels, dx-21, dy-21]
inp_dim = 81*81

model_reg = AE(inp_dim, hidden_size, latent_dim, no_layers, Sin()).to(device)
model_base = AE(inp_dim, hidden_size, latent_dim, no_layers, Sin()).to(device)

#model_reg.load_state_dict(torch.load(path+'model_reg'+name, map_location=torch.device('cpu'))["model"])
#model_base.load_state_dict(torch.load(path+'model_reg'+name, map_location=torch.device('cpu'))["model"])

model_reg.load_state_dict(torch.load(path+'model_reg'+name, map_location=torch.device('cpu')))
model_base.load_state_dict(torch.load(path+'model_base'+name, map_location=torch.device('cpu')))
#model_reg.eval()
#model_base.eval()


deg_quad = 80
u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
x = np.linspace(-1,1,96)
X_p = u_ob.data_axes([x,x]).T

# now get PSNR vs perturbation rate  plots
noise_order = torch.linspace(1e-3,100* 1e-3, 100).to('cpu')

all_PSNR_bAE = torch.tensor([])
all_PSNR_rAE = torch.tensor([])

all_PSNR_bAE = []
all_PSNR_rAE = []
for i in range(len(noise_order)):
    testCoeffs_pert_Fplt = testCoeffs +  noise_order[i] * torch.rand(testCoeffs.shape)
    testCoeffs_pert_Fplt = testCoeffs_pert_Fplt.to('cpu')
    ImagesFromPerturbedTestCoeffs_Fplt = torch.matmul(X_p.float(), testCoeffs_pert_Fplt.float().squeeze(1).T).T
    ImagesFromPerturbedTestCoeffs_Fplt = ImagesFromPerturbedTestCoeffs_Fplt.reshape(Analys_size,1,96,96)
    rec_rAE_test_perturbedCoeffs_Fplt = model_reg(testCoeffs_pert_Fplt.float()).view(testCoeffs_pert_Fplt.shape)
    rec_bAE_test_perturbedCoeffs_Fplt = model_base(testCoeffs_pert_Fplt.float()).view(testCoeffs_pert_Fplt.shape)
    rec_rAE_test_perturbedCoeffs_Fplt = torch.tensor(rec_rAE_test_perturbedCoeffs_Fplt, requires_grad=False)
    rec_bAE_test_perturbedCoeffs_Fplt = torch.tensor(rec_bAE_test_perturbedCoeffs_Fplt, requires_grad=False)
    reconReg_test_perturbedCoeffs_Fplt = torch.matmul(X_p.float(), rec_rAE_test_perturbedCoeffs_Fplt.squeeze(1).T).T
    reconBase_test_perturbedCoeffs_Fplt = torch.matmul(X_p.float(), rec_bAE_test_perturbedCoeffs_Fplt.squeeze(1).T).T
    reconReg_test_perturbedCoeffs_Fplt = reconReg_test_perturbedCoeffs_Fplt.reshape(Analys_size,1,96,96)
    reconBase_test_perturbedCoeffs_Fplt = reconBase_test_perturbedCoeffs_Fplt.reshape(Analys_size,1,96,96)
    #PSNR for Images from perturbedcoeffs in train dataset
    psnr_rAE_test_perturbedCoeffs_Fplt = batch_psnr(reconReg_test_perturbedCoeffs_Fplt.reshape(testImages[:500].shape), testImages[:500])
    psnr_bAE_test_perturbedCoeffs_Fplt = batch_psnr(reconBase_test_perturbedCoeffs_Fplt.reshape(testImages[:500].shape), testImages[:500])
    print("PSNR rAE: %.2f, bAE: %.2f" % (psnr_rAE_test_perturbedCoeffs_Fplt, psnr_bAE_test_perturbedCoeffs_Fplt))

    all_PSNR_bAE.append(psnr_bAE_test_perturbedCoeffs_Fplt)
    all_PSNR_rAE.append(psnr_rAE_test_perturbedCoeffs_Fplt) 
    


all_PSNR_bAE = torch.tensor(all_PSNR_bAE)
all_PSNR_rAE = torch.tensor(all_PSNR_rAE)
torch.save(all_PSNR_bAE, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/PSNR_list_comparisons/all_PSNR_bAE_p001L18.pt')
torch.save(all_PSNR_rAE, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/PSNR_list_comparisons/all_PSNR_rAE_p001L18.pt')