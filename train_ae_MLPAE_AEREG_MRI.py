import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

from random import seed
from random import randint
#from FashionMNIST5LayersTrials.regularisers import computeC1Loss_upd
from regularisers import computeC1Loss_upd
seed(1)

#from swd import swd

from regularisers import sampleNodes, computeC1Loss, sampleChebyshevNodes, sampleLegendreNodes, barycenterSubsampleLegendreNodes

from barycenter_subsampling import get_convergent_barycenters
from models import AE
from datasets import  getDataset
import copy

import os
import re
import ot
import wandb
from datasets import  getDataset

#import minterpy as mp

os.environ['WANDB_API_KEY'] = 'e1a3b85ef17f1f7e3a683f4dd0d1fcbacb4668b1'
wandb.login()

#import sinkhorn_pointcloud as spc
# Sinkhorn parameters
epsilon = 0.01
niter = 100

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
from torch.autograd import grad
def loss_grad_std_full(loss, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float96,device=device)
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if(m == 0):
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]        
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]        
            grad_ = torch.cat((grad_,w.view(-1), b))
            
    return torch.std(grad_)

def train(trainImagesInBatches, testImagesInBatches, no_channels, dx, dy, no_epochs=2, TDA=0.4, reco_loss='mse', latent_dim=10, 
          hidden_size=1024, no_layers=3, activation = F.relu, lr = 3e-4, alpha=1., bl=False, 
          seed = 2342, train_base_model=False, no_samples=5, deg_poly=21,
          reg_nodes_sampling="legendre_exp", no_val_samples = 10, use_guidance = True,
          enable_wandb=True, wandb_project=None, wandb_entity=None, weight_jac = False):


    wass_outputs = []
    wass_outputs_val = []

    if (reg_nodes_sampling == 'legendre'):
        points = np.polynomial.legendre.leggauss(deg_poly)[0][::-1]
        
        weights = np.polynomial.legendre.leggauss(deg_poly)[1][::-1]


    if (reg_nodes_sampling == 'bary_legendre'):
        points = np.polynomial.legendre.leggauss(deg_poly)[0][::-1]
        
        weights = np.polynomial.legendre.leggauss(deg_poly)[1][::-1]
        #print('points.shape', points.shape[0])
        oneD_LegendreGrid = points.reshape(points.shape[0],1)
        sweeing_radius = 0.15
        leg_bary, leg_indices = get_convergent_barycenters(oneD_LegendreGrid, oneD_LegendreGrid[oneD_LegendreGrid.shape[0]//2],sweeing_radius)

        #print('leg_bary.shape', leg_bary.shape)
        no_samples = leg_bary.shape[0]
        z1, z2 = np.meshgrid(leg_bary, leg_bary) 

        z1 = z1.flatten()
        z2 = z2.flatten()

        oneDSubsampledLeg_grid = np.column_stack((z1,z2))
        #print("check bary legendre")

        #print('oneDSubsampledLeg_grid.shape', oneDSubsampledLeg_grid.shape)



    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    set_seed(2342)

    no_channels = 1
    dx = 96
    dy = 96
    inp_dim = [no_channels, dx, dy]


    model_reg = AE(inp_dim, hidden_size, latent_dim, 
                       no_layers, activation).to(device) # regularised autoencoder

    if train_base_model:
        model = AE(inp_dim, hidden_size, latent_dim, 
                       no_layers, activation).to(device) # baseline autoencoder
        model = copy.deepcopy(model_reg)


    global_step = 0
    cond_step = 0
    optimizer = torch.optim.Adam(model_reg.parameters(), lr=lr)
    lamb = 1.
    if train_base_model:
        optimizer_base = torch.optim.Adam(model.parameters(), lr=lr)
    
    #arrays for holding loss
    loss_arr_reg = []
    loss_arr_reco = []
    loss_arr_base = []
    
    loss_arr_val_reco = []
    loss_arr_val_base = []
    
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    if enable_wandb:
        wandb.watch(model_reg, log="all")
        if train_base_model:
            wandb.watch(model, log="all")

    #Jac_val_pts = torch.FloatTensor(np.random.uniform(-1,1,size=(no_val_samples, latent_dim))).to(device)
    

    trainImagesInBatches = trainImagesInBatches[:int(trainImagesInBatches.shape[0]*TDA)]
    testImagesInBatches = testImagesInBatches[:int(testImagesInBatches.shape[0]*TDA)]

    print('Again image_batches_trn.shape',trainImagesInBatches.shape)
    print('Again image_batches_test.shape',testImagesInBatches.shape)


    for epoch in tqdm(range(no_epochs)):
        
        loss_full = []
        loss_rec = []
        loss_rec_base = []
        loss_c1 = []
        print('Epoch : '+str(epoch)+ 'started')
        inum = 0
        #for inum, batch_x in enumerate(train_loader):
        #inum = 0
        for batch_x in trainImagesInBatches:    
            inum = inum+1

            batch_x = batch_x.float()
            global_step += 1
            loss_C1 = torch.FloatTensor([0.]).to(device) 
            # plain reconstruction using AE
            batch_x = batch_x.to(device)

            reconstruction = model_reg(batch_x)
            reconstruction = reconstruction.view(batch_x.shape)



            loss_reconstruction = F.mse_loss(reconstruction, batch_x)

            nodes_subsample_np, weights_subsample_np = sampleLegendreNodes(no_samples, latent_dim, weight_jac, points, weights, n=deg_poly)
            nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)
            #weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)




            #loss_C1, Jac = computeC1Loss(nodes_subsample, model_reg, device, guidanceTerm = use_guidance) # guidance term
            loss_C1, Jac = computeC1Loss_upd(nodes_subsample, model_reg, device, guidanceTerm = use_guidance) # guidance term
            #print('Jac.shape', Jac.shape)

            loss = (1.- alpha)*loss_reconstruction + alpha*loss_C1
            loss_full.append(float(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            reco_base = model(batch_x).view(batch_x.size())
            loss_base = F.mse_loss(reco_base, batch_x)
            loss_rec_base.append(float(loss_base.item()))
            optimizer_base.zero_grad()
            loss_base.backward()
            optimizer_base.step()

            loss_rec.append(float(loss_reconstruction.item()))
            loss_c1.append(float(loss_C1.item())/batch_x.shape[0])
            

        loss_arr_reg.append(torch.Tensor([sum(loss_full)/len(loss_full)]))
        loss_arr_reco.append(torch.Tensor([sum(loss_rec)/len(loss_rec)]))
        loss_arr_base.append(torch.Tensor([sum(loss_rec_base)/len(loss_rec_base)]))
        
        
        print()
        print("[%d] rAE loss = %.4e, rAE reconstruction loss = %.4e" % (epoch, loss, loss_reconstruction))
        print()

 
        
    if train_base_model:

        path = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'
        #path = './output/MRT_full/test_run_saving/'
        os.makedirs(path, exist_ok=True)
        name = '_'+reg_nodes_sampling+'_'+'_'+str(TDA)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)+'_'+str(no_samples)
        torch.save(loss_arr_reg, path+'/loss_arr_reg_MRI_TDA'+name)
        torch.save(loss_arr_reco, path+'/loss_arr_reco_MRI_TDA'+name)
        torch.save(loss_arr_base, path+'/loss_arr_base_MRI_TDA'+name)
        torch.save(loss_arr_val_reco, path+'/loss_arr_val_reco_MRI_TDA'+name)
        torch.save(loss_arr_val_base, path+'/loss_arr_val_MRI_MRI_TDA'+name)
        torch.save(model.state_dict(), path+'/model_base_MRI_TDA'+name)
        torch.save(model_reg.state_dict(), path+'/model_reg_MRI_TDA'+name)

        #torch.save(model, path+'/model_base_old_try'+name)
        #torch.save(model_reg, path+'/model_reg_old_try'+name)        
        return model, model_reg, loss_arr_reg, loss_arr_reco, loss_arr_base, loss_arr_val_reco, loss_arr_val_base
    else:
        return model_reg, loss_arr_reg, loss_arr_reco