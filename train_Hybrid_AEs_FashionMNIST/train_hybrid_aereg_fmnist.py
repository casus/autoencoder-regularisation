import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

from regularisers_without_vegas_fmnist import sampleNodes, computeC1Loss, sampleChebyshevNodes, sampleLegendreNodes, computeC1Loss_upd
from models import AE
import copy
from jmp_solver1.diffeomorphisms import hyper_rect
import jmp_solver1.surrogates

import os
import re
import ot
from scipy import interpolate
#import wandb



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



def train(image_batches_trn, image_batches_test, coeffs_saved_trn, coeffs_saved_test, no_epochs=60, reco_loss='mse', latent_dim=20, 
          hidden_size=1000, no_layers=5, activation = F.relu, lr = 1e-4, alpha=1e-3, bl=False, 
          seed = 2342, train_base_model=True, no_samples=5, deg_poly=20,
          reg_nodes_sampling="legendre", no_val_samples = 10, HybridPolyDegree = 20, use_guidance = True, train_set_size=0.8,
          enable_wandb=False, wandb_project=None, wandb_entity=None):



    weight_jac = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    


    set_seed(2342)
    

    deg_quad = HybridPolyDegree 
    points = np.polynomial.legendre.leggauss(deg_poly)[0][::-1]    
    weights = np.polynomial.legendre.leggauss(deg_poly)[1][::-1]


    inp_dim = (deg_quad+1)*(deg_quad+1)
    model_reg = AE(inp_dim, hidden_size, latent_dim, 
                    no_layers, activation).to(device) # regularised autoencoder
    if train_base_model:
        model = AE(inp_dim, hidden_size, latent_dim, 
                    no_layers, activation).to(device) # baseline autoencoder
        model = copy.deepcopy(model_reg)


    global_step = 0
        
    optimizer = torch.optim.Adam(model_reg.parameters(), lr=lr, amsgrad=True)
    
    lamb = 1.
    if train_base_model:
        optimizer_base = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    

    loss_arr_reg = []
    loss_arr_reco = []
    loss_arr_base = []
    
    loss_arr_val_reco = []
    loss_arr_val_base = []
    

    Jac_val_pts = torch.FloatTensor(np.random.uniform(-1,1,size=(no_val_samples, latent_dim))).to(device)
    

    u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
    x = np.linspace(-1,1,32)
    x = torch.tensor(x)
    x = x.float()
    X_p = (u_ob.data_axes([x,x]).T)
    X_p = (X_p.float()).to(device)

    image_batches_trn = image_batches_trn[:int(image_batches_trn.shape[0]*train_set_size)]
    image_batches_test = image_batches_test[:int(image_batches_test.shape[0]*train_set_size)]
    
    coeffs_saved_trn = coeffs_saved_trn[:int(coeffs_saved_trn.shape[0]*train_set_size)]
    coeffs_saved_test = coeffs_saved_test[:int(coeffs_saved_test.shape[0]*train_set_size)]


    for epoch in tqdm(range(no_epochs)):
        loss_full = []
        loss_rec = []
        loss_rec_base = []
        loss_c1 = []
        inum = 0
        for batch_x in coeffs_saved_trn:    
            inum = inum + 1
            global_step += 1
            loss_C1 = torch.FloatTensor([0.]).to(device)
            batch_x = batch_x.float().to(device)

            reconstruction = model_reg(batch_x)
            image_batches_trnp = image_batches_trn[inum-1]
            reconstructionH = (torch.matmul(X_p, reconstruction.squeeze(1).T).T).reshape(reconstruction.shape[0], 1, 32, 32)
            if reco_loss == 'mse':
                loss_reconstruction = F.mse_loss(reconstructionH, image_batches_trnp)


            if(reg_nodes_sampling == 'chebyshev'):
                nodes_subsample_np, weights_subsample_np = sampleChebyshevNodes(no_samples, latent_dim, weight_jac, n=deg_poly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)
                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
            elif(reg_nodes_sampling == 'legendre'): 
                nodes_subsample_np, weights_subsample_np = sampleLegendreNodes(no_samples, latent_dim, weight_jac, points, weights, n=deg_poly)
                
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)

                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
            elif(reg_nodes_sampling == 'random'):
                nodes_subsample = torch.FloatTensor(no_samples, latent_dim).uniform_(-1, 1)
            elif(reg_nodes_sampling == 'trainingData'):
                nodes_subsample = model_reg.encoder(batch_x[0:no_samples, :]).detach()

            loss_C1, Jac = computeC1Loss_upd(nodes_subsample, model_reg, device, guidanceTerm = use_guidance) # guidance term
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()

            # train baseline autoencoder
            if train_base_model:
                reco_base = model(batch_x)#.view(batch_x.size())
                reco_base = (torch.matmul(X_p, reco_base.squeeze(1).T).T).reshape(reco_base.shape[0], 1, 32, 32)
                loss_base = F.mse_loss(reco_base, image_batches_trnp)
                #loss_base = F.mse_loss(reco_base, batch_x)
                loss_rec_base.append(float(loss_base.item()))
                optimizer_base.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                loss_base.backward()
                optimizer_base.step()

            loss_rec.append(float(loss_reconstruction.item()))
            loss_c1.append(float(loss_C1.item())/batch_x.shape[0])
            
            if (global_step % 2) == 0:
                _, Jac = computeC1Loss(Jac_val_pts, model_reg, device)
                magicNo, _, _, _ = Jac.shape
                Jac_m = torch.mean(Jac, axis=[0,2]) * magicNo
                
                rank_Jacm = float(torch.matrix_rank(Jac_m, tol=1e-1).cpu().detach().numpy())
                cond_Jacm = np.linalg.cond(Jac_m.detach().cpu().numpy())
                
                    
                if train_base_model:
                    _, Jac = computeC1Loss(Jac_val_pts, model, device)
                    magicNo, _, _, _ = Jac.shape
                    Jac_m = torch.mean(Jac, axis=[0,2]) * magicNo

                    rank_Jacm = float(torch.matrix_rank(Jac_m, tol=1e-1).cpu().detach().numpy())
                    cond_Jacm = np.linalg.cond(Jac_m.detach().cpu().numpy())

                    
        loss_arr_reg.append(torch.Tensor([sum(loss_full)/len(loss_full)]))
        loss_arr_reco.append(torch.Tensor([sum(loss_rec)/len(loss_rec)]))
        if train_base_model:
            loss_arr_base.append(torch.Tensor([sum(loss_rec_base)/len(loss_rec_base)]))
        
        
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
            for batch_val in coeffs_saved_test:
                inum_ = inum_ + 1
                val_step += 1
                loss_C1_val = torch.FloatTensor([0.]).to(device)
                batch_val = batch_val.float()
                batch_val = batch_val.to('cuda')
                reconstruction_val = model_reg(batch_val)
                image_batches_testp = image_batches_test[inum_-1]
                reconstructionH_val = (torch.matmul(X_p, reconstruction_val.squeeze(1).T).T).reshape(reconstruction_val.shape[0], 1, 32, 32)

                if reco_loss == 'mse':
                    loss_reconstruction_val = F.mse_loss(reconstructionH_val, image_batches_testp)
                    
                tmp_loss_list.append(float(loss_reconstruction_val.item()))


                    
                if train_base_model:
                    reco_base = model(batch_val)#.view(batch_val.size())

                    reco_base = (torch.matmul(X_p, reco_base.squeeze(1).T).T).reshape(reco_base.shape[0], 1, 32, 32)
                    loss_base_val_ = F.mse_loss(reco_base, image_batches_testp)
                    tmp_base_list.append(float(loss_base_val_.item()))

        loss_rec_val.append(sum(tmp_loss_list)/len(tmp_loss_list))
        loss_arr_val_reco.append(torch.Tensor([sum(tmp_loss_list)/len(tmp_loss_list)]))
        if train_base_model:
            loss_base_val.append(sum(tmp_base_list)/len(tmp_base_list))
            loss_arr_val_base.append(torch.Tensor([sum(tmp_base_list)/len(tmp_base_list)]))
        
    path = './models_saved/'
    os.makedirs(path, exist_ok=True)
    name = '_'+reg_nodes_sampling+'_'+str(train_set_size)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(train_set_size)
    torch.save(loss_arr_reg, path+'/loss_arr_regLSTQS'+str(deg_quad)+''+name)
    torch.save(loss_arr_reco, path+'/loss_arr_recoLSTQS'+str(deg_quad)+''+name)
    torch.save(loss_arr_base, path+'/loss_arr_baseLSTQS'+str(deg_quad)+''+name)
    torch.save(loss_arr_val_reco, path+'/loss_arr_val_recoLSTQS'+str(deg_quad)+''+name)
    torch.save(loss_arr_val_base, path+'/loss_arr_val_baseLSTQS'+str(deg_quad)+''+name)
    torch.save(model.state_dict(), path+'/model_baseLSTQS'+str(deg_quad)+''+name)
    torch.save(model_reg.state_dict(), path+'/model_regLSTQS'+str(deg_quad)+''+name)

    #torch.save(model, path+'/model_base_old_try'+name)
    #torch.save(model_reg, path+'/model_reg_old_try'+name)
    return model, model_reg, loss_arr_reg, loss_arr_reco, loss_arr_base, loss_arr_val_reco, loss_arr_val_base
        