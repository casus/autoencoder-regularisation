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

from regularisers import sampleNodes, computeC1Loss, sampleChebyshevNodes, sampleLegendreNodes
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
    grad_ = torch.zeros((0), dtype=torch.float32,device=device)
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
          seed = 2342, train_base_model=False, no_samples=5, deg_poly=10,
          reg_nodes_sampling="legendre_exp", no_val_samples = 10, use_guidance = True,
          enable_wandb=True, wandb_project=None, wandb_entity=None, weight_jac = False):



    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(2342)


    # creating Legendre points in 1D 
    points = np.polynomial.legendre.leggauss(deg_poly)[0][::-1]
    weights = np.polynomial.legendre.leggauss(deg_poly)[1][::-1]

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
    
    Jac_val_pts = torch.FloatTensor(np.random.uniform(-1,1,size=(no_val_samples, latent_dim))).to(device)


    trainImagesInBatches = trainImagesInBatches[:int(trainImagesInBatches.shape[0]*TDA)]
    testImagesInBatches = testImagesInBatches[:int(testImagesInBatches.shape[0]*TDA)]

    print('image_batches_trn.shape',trainImagesInBatches.shape)
    print('image_batches_test.shape',testImagesInBatches.shape)

    for epoch in tqdm(range(no_epochs)):
        
        loss_full = []
        loss_rec = []
        loss_rec_base = []
        loss_c1 = []
        print('Epoch : '+str(epoch)+ 'started')
        inum = 0

        for batch_x in trainImagesInBatches:    
            inum = inum+1

            batch_x = batch_x.float()
            global_step += 1
            loss_C1 = torch.FloatTensor([0.]).to(device) 

            batch_x = batch_x.to(device)
            #print('batch_x.shape', batch_x.shape)
            reconstruction = model_reg(batch_x)
            reconstruction = reconstruction.view(batch_x.shape)

            if reco_loss == 'mse':
                loss_reconstruction = F.mse_loss(reconstruction, batch_x)
            if reco_loss == 'wasserstein':
                loss_reconstruction = swd(batch_x, reconstruction, device="cuda")
                #loss_reconstruction = torch.sum(result_wass)



            if(reg_nodes_sampling == 'chebyshev'):
                nodes_subsample_np, weights_subsample_np = sampleChebyshevNodes(no_samples, latent_dim, weight_jac, n=deg_poly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)
                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
            elif(reg_nodes_sampling == 'legendre'): 
                nodes_subsample_np, weights_subsample_np = sampleLegendreNodes(no_samples, latent_dim, weight_jac, points, weights, n=deg_poly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)

                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
            elif(reg_nodes_sampling == 'legendre_exp'): 
                nodes_subsample_np, weights_subsample_np = sampleLegendreNodes(no_samples, latent_dim, weight_jac, n=deg_poly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)
                nodes_subsample = (nodes_subsample+1)/2
                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
                #print('nodes_subsample.max()',nodes_subsample.max())
                #print('nodes_subsample.min()',nodes_subsample.min())

            elif(reg_nodes_sampling == 'random'):
                nodes_subsample = torch.FloatTensor(no_samples, latent_dim).uniform_(-1, 1)
            elif(reg_nodes_sampling == 'trainingData'):
                nodes_subsample = model_reg.encoder(batch_x[0:no_samples, :]).detach()




            #loss_C1, Jac = computeC1Loss(nodes_subsample, model_reg, device, guidanceTerm = use_guidance) # guidance term
            loss_C1, Jac = computeC1Loss_upd(nodes_subsample, model_reg, device, guidanceTerm = use_guidance) # guidance term
            #print('Jac.shape', Jac.shape)
            if bl:
                with torch.no_grad():
                    stdr = loss_grad_std_full(loss_reconstruction, model_reg)
                    stdb = loss_grad_std_full(loss_C1, model_reg)
                    lamb_hat = stdr/stdb
                    alpha = 0.5
                    lamb     = (1.-alpha)*lamb + alpha*lamb_hat
                loss = loss_reconstruction + lamb*loss_C1
            else:
                loss = (1.- alpha)*loss_reconstruction + alpha*loss_C1
            loss_full.append(float(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train baseline autoencoder
            if train_base_model:
                reco_base = model(batch_x).view(batch_x.size())
                #batch_xHb = torch.matmul(R_matrix, batch_x.squeeze(1).T).T
                #reco_baseHb = torch.matmul(R_matrix, reco_base.squeeze(1).T).T
                loss_base = F.mse_loss(reco_base, batch_x)
                loss_rec_base.append(float(loss_base.item()))
                optimizer_base.zero_grad()
                loss_base.backward()
                optimizer_base.step()

            loss_rec.append(float(loss_reconstruction.item()))
            loss_c1.append(float(loss_C1.item())/batch_x.shape[0])
            
            if (global_step % 100) == 0:
                _, Jac = computeC1Loss(Jac_val_pts, model_reg, device)
                magicNo, _, _, _ = Jac.shape
                Jac_m = torch.mean(Jac, axis=[0,2]) * magicNo
                
                rank_Jacm = float(torch.matrix_rank(Jac_m, tol=1e-1).cpu().detach().numpy())
                cond_Jacm = np.linalg.cond(Jac_m.detach().cpu().numpy())
                
                if enable_wandb:
                    wandb.log({'rAE-rank': rank_Jacm })
                    wandb.log({'rAE-cond': cond_Jacm })
                    cond_step += 1
                    plt.imshow(Jac_m.squeeze().detach().cpu())
                    plt.title('Rec µ(Jacobian), step %d' % (global_step))
                    plt.colorbar()
                    wandb.log({"rAE-meanJacobian": plt})
                    plt.close()

                else:
                    print("[%d] rAE rank = %d, cond = %.4e" % (epoch, rank_Jacm, cond_Jacm))
                    
                if train_base_model:
                    _, Jac = computeC1Loss(Jac_val_pts, model, device)
                    magicNo, _, _, _ = Jac.shape
                    Jac_m = torch.mean(Jac, axis=[0,2]) * magicNo

                    rank_Jacm = float(torch.matrix_rank(Jac_m, tol=1e-1).cpu().detach().numpy())
                    cond_Jacm = np.linalg.cond(Jac_m.detach().cpu().numpy())

                    if enable_wandb:
                        wandb.log({'bAE-rank': rank_Jacm })
                        wandb.log({'bAE-cond': cond_Jacm })
                        cond_step += 1

                        plt.imshow(Jac_m.squeeze().detach().cpu())
                        plt.title('Base µ(Jacobian), step %d' % (global_step))
                        plt.colorbar()
                        wandb.log({"bAE-meanJacobian": plt})
                        plt.close()
                    
        loss_arr_reg.append(torch.Tensor([sum(loss_full)/len(loss_full)]))
        loss_arr_reco.append(torch.Tensor([sum(loss_rec)/len(loss_rec)]))
        if train_base_model:
            loss_arr_base.append(torch.Tensor([sum(loss_rec_base)/len(loss_rec_base)]))
        
        if enable_wandb:
            wandb.log({'rAE-loss_reco': sum(loss_rec)/len(loss_rec),
                        'rAE-loss_C1': sum(loss_c1)/len(loss_c1)})
            if train_base_model:
                wandb.log({'bAE-loss_reco': sum(loss_rec_base)/len(loss_rec_base)})
        
        print()
        print("[%d] rAE loss = %.4e, rAE reconstruction loss = %.4e" % (epoch, loss, loss_reconstruction))
        print()

        loss_rec_val = []
        loss_base_val = []

        val_step = 0
        with torch.no_grad():
            tmp_loss_list = []
            tmp_base_list = []
            inum_ = 0
            #for inum_, batch_val in enumerate(test_loader):
            #inum_ = 0
            for batch_val in testImagesInBatches:
                inum_ = inum_ + 1
                batch_val = batch_val[0]
                batch_val = batch_val.float()
                val_step += 1
                loss_C1_val = torch.FloatTensor([0.]).to(device)
                batch_val = batch_val.to(device)
                reconstruction_val = model_reg(batch_val)
                reconstruction_val = reconstruction_val.view(batch_val.shape)

                if reco_loss == 'mse':
                    loss_reconstruction_val = F.mse_loss(reconstruction_val, batch_val)
                if reco_loss == 'wasserstein':
                    loss_reconstruction_val = swd(batch_val, reconstruction_val, device="cuda")

                tmp_loss_list.append(float(loss_reconstruction_val.item()))
                
                kk = randint(0, 100)

                #print('inum',inum_)
                if enable_wandb:
                    wandb.log({'rAE-loss_reco_val': float(loss_reconstruction_val.item())})
                    if (inum_ == 2):
                        #print(reconstruction_val[kk][0].shape)
                        #plt.imshow(reconstruction_val[0,:].reshape(no_channels,dx,dy).squeeze(0).detach().cpu().numpy())
                        plt.imshow(reconstruction_val[kk].reshape(32,32).detach().cpu().numpy())
                        plt.title('Wasserstein Reconstruction, step %d' % (epoch))
                        plt.colorbar()

                        wandb.log({"rAE-reco": plt})
                        plt.close()
                
                if train_base_model:
                    reco_val_base = model(batch_val).view(batch_val.size())
                    #batch_valHb = torch.matmul(R_matrix, batch_val.squeeze(1).T).T
                    #reco_base_valHb = torch.matmul(R_matrix, reco_val_base.squeeze(1).T).T
                    loss_base_val_ = F.mse_loss(reco_val_base, batch_val)
                    tmp_base_list.append(float(loss_base_val_.item()))
                    if enable_wandb:
                        wandb.log({'bAE-loss_reco_val': float(loss_base_val_.item())}) 
                        if (inum_ == 2):
                            #plt.imshow(reco_base[0,:].reshape(no_channels, dx, dy).squeeze(0).detach().cpu().numpy())
                            plt.imshow(reco_val_base[kk].reshape(32,32).detach().cpu().numpy()) 
                            plt.title('Base Reconstruction, step %d' % (epoch))
                            plt.colorbar()


                            wandb.log({"bAE-reco": plt})
                            plt.close()

        loss_rec_val.append(sum(tmp_loss_list)/len(tmp_loss_list))
        loss_arr_val_reco.append(torch.Tensor([sum(tmp_loss_list)/len(tmp_loss_list)]))
        if train_base_model:
            loss_base_val.append(sum(tmp_base_list)/len(tmp_base_list))
            loss_arr_val_base.append(torch.Tensor([sum(tmp_base_list)/len(tmp_base_list)]))
        if enable_wandb:
            wandb.log({'rAE-loss_reco_val_min': min(loss_rec_val)})
            wandb.log({'bAE-loss_reco_val_min': min(loss_base_val)})
            wandb.log({'epoch': epoch }) 
        
    if train_base_model:

        path = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'
        #path = './output/MRT_full/test_run_saving/'
        os.makedirs(path, exist_ok=True)
        name = '_'+reg_nodes_sampling+'_'+'_'+str(TDA)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(TDA)
        torch.save(loss_arr_reg, path+'/loss_arr_reg_TDA'+name)
        torch.save(loss_arr_reco, path+'/loss_arr_reco_TDA'+name)
        torch.save(loss_arr_base, path+'/loss_arr_base_TDA'+name)
        torch.save(loss_arr_val_reco, path+'/loss_arr_val_reco_TDA'+name)
        torch.save(loss_arr_val_base, path+'/loss_arr_val_base_TDA'+name)
        torch.save(model.state_dict(), path+'/model_baseUpd_TDA'+name)
        torch.save(model_reg.state_dict(), path+'/model_regUpd_TDA'+name)

        #torch.save(model, path+'/model_base_old_try'+name)
        #torch.save(model_reg, path+'/model_reg_old_try'+name)        
        return model, model_reg, loss_arr_reg, loss_arr_reco, loss_arr_base, loss_arr_val_reco, loss_arr_val_base
    else:
        return model_reg, loss_arr_reg, loss_arr_reco