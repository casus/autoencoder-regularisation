import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

from regularisers_without_vegas import sampleNodes, computeC1Loss, sampleChebyshevNodes, sampleLegendreNodes
from models import AE
import copy
#from layers import SinkhornDistance
#import sinkhorn_pointcloud as spc
# Sinkhorn parameters
#epsilon = 0.01
#niter = 100
from jmp_solver1.diffeomorphisms import hyper_rect
import jmp_solver1.surrogates

#from jmp_solver.diffeomorphisms import hyper_rect
#import jmp_solver.surrogates 

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



def train(train_loader, test_loader, no_epochs=60, reco_loss='mse', latent_dim=20, 
          hidden_size=1000, no_layers=5, activation = F.relu, lr = 1e-4, alpha=1e-3, bl=False, 
          seed = 2342, train_base_model=True, no_samples=5, deg_poly=20,
          reg_nodes_sampling="legendre", no_val_samples = 10, use_guidance = True, train_set_size=0.8,
          enable_wandb=False, wandb_project=None, wandb_entity=None):


    #sinkhorn = SinkhornDistance(eps=0.0000000001, max_iter=100)
    wass_outputs = []
    wass_outputs_val = []

    weight_jac = False
    if enable_wandb:
        import wandb
        wandb.init(project=wandb_project, entity=wandb_entity)
        wandb.config = {
            "no_layers": no_layers,
            "hidden_units": hidden_size,
            "deg_poly": deg_poly,
            "reg_nodes_sampling": reg_nodes_sampling,
            "alpha": alpha,
            "lr": lr,
            "diag": use_guidance
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    '''no_channels = 1
    dx, dy = (train_loader.dataset.__getitem__(1).shape)'''
    print('(train_loader.dataset.__getitem__(1).shape)',(train_loader.dataset.__getitem__(1).shape))
    no_channels, dx, dy = (train_loader.dataset.__getitem__(1).shape)


    set_seed(2342)

    #inp_dim = [no_channels, dx-21, dy-21]
    #inp_dim = [no_channels, deg_leg, deg_leg]
    inp_dim = 81*81
    model_reg = AE(inp_dim, hidden_size, latent_dim, 
                       no_layers, activation).to(device) # regularised autoencoder
    if train_base_model:
        model = AE(inp_dim, hidden_size, latent_dim, 
                       no_layers, activation).to(device) # baseline autoencoder
        model = copy.deepcopy(model_reg)
                                                                                                             
    global_step = 0
    cond_step = 0

        
    optimizer = torch.optim.Adam(model_reg.parameters(), lr=lr, amsgrad=True)
    
    lamb = 1.
    if train_base_model:
        optimizer_base = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    
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

    Jac_val_pts = torch.FloatTensor(np.random.uniform(-1,1,size=(no_val_samples, latent_dim))).to(device)
    
    deg_quad = 80
    u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
    x = np.linspace(-1,1,96)
    x = torch.tensor(x)
    x = x.float()
    X_p = (u_ob.data_axes([x,x]).T)
    X_p = (X_p.float()).to(device)

    batch_size_cfs = 200
    coeffs_saved_trn = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/trainDataRK_coeffs.pt').to(device)
    image_batches_trn = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/trainDataSet.pt').to(device)
    image_batches_trn = image_batches_trn.reshape(int(image_batches_trn.shape[0]/batch_size_cfs), batch_size_cfs, 1, 96,96)
    coeffs_saved_trn = coeffs_saved_trn.reshape(int(coeffs_saved_trn.shape[0]/batch_size_cfs), batch_size_cfs, coeffs_saved_trn.shape[1]).unsqueeze(2) 


    coeffs_saved_test = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/testDataRK_coeffs.pt').to(device)
    image_batches_test = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/testDataSet.pt').to(device)
    image_batches_test = image_batches_test[:11200]
    image_batches_test = image_batches_test.reshape(int(image_batches_test.shape[0]/batch_size_cfs), batch_size_cfs, 1, 96,96)
    coeffs_saved_test = coeffs_saved_test[:11200]
    coeffs_saved_test = coeffs_saved_test.reshape(int(coeffs_saved_test.shape[0]/batch_size_cfs), batch_size_cfs, coeffs_saved_test.shape[1]).unsqueeze(2) 

    print('coeffs_saved_trn.shape',coeffs_saved_trn.shape)

    print('image_batches_trn.shape',image_batches_trn.shape)

    print('coeffs_saved_test.shape',coeffs_saved_test.shape)

    print('image_batches_test.shape',image_batches_test.shape)

    image_batches_trn = image_batches_trn[:int(image_batches_trn.shape[0]*train_set_size)]
    image_batches_test = image_batches_test[:int(image_batches_test.shape[0]*train_set_size)]
    
    coeffs_saved_trn = coeffs_saved_trn[:int(coeffs_saved_trn.shape[0]*train_set_size)]
    coeffs_saved_test = coeffs_saved_test[:int(coeffs_saved_test.shape[0]*train_set_size)]


    print('coeffs_saved_trn.shape',coeffs_saved_trn.shape)

    print('image_batches_trn.shape',image_batches_trn.shape)

    print('coeffs_saved_test.shape',coeffs_saved_test.shape)

    print('image_batches_test.shape',image_batches_test.shape)



    for epoch in tqdm(range(no_epochs)):
        loss_full = []
        loss_rec = []
        loss_rec_base = []
        loss_c1 = []

        #for inum, batch_x in enumerate(train_loader):
        inum = 0
        for batch_x in coeffs_saved_trn:    
            inum = inum + 1
            global_step += 1
            loss_C1 = torch.FloatTensor([0.]).to(device)
            batch_x = batch_x.float().to(device)
            #batch_x = torch.FloatTensor(batch_x)
            #batch_x = batch_x.to(device)

            reconstruction = model_reg(batch_x)
            #print('before view', reconstruction.shape)
            #reconstruction = reconstruction.view(batch_x.shape)
            #print('inum',inum)
            #print('after view')
            #print('reconstruction.shape',reconstruction.shape)
            image_batches_trnp = image_batches_trn[inum-1]
            reconstructionH = (torch.matmul(X_p, reconstruction.squeeze(1).T).T).reshape(reconstruction.shape[0], 1, 96, 96)
            if reco_loss == 'mse':
                loss_reconstruction = F.mse_loss(reconstructionH, image_batches_trnp)


            if(reg_nodes_sampling == 'chebyshev'):
                nodes_subsample_np, weights_subsample_np = sampleChebyshevNodes(no_samples, latent_dim, weight_jac, n=deg_poly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)
                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
            elif(reg_nodes_sampling == 'legendre'): 
                nodes_subsample_np, weights_subsample_np = sampleLegendreNodes(no_samples, latent_dim, weight_jac, n=deg_poly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)

                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
            elif(reg_nodes_sampling == 'random'):
                nodes_subsample = torch.FloatTensor(no_samples, latent_dim).uniform_(-1, 1)
            elif(reg_nodes_sampling == 'trainingData'):
                nodes_subsample = model_reg.encoder(batch_x[0:no_samples, :]).detach()

            loss_C1, Jac = computeC1Loss(nodes_subsample, model_reg, device, guidanceTerm = use_guidance) # guidance term
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
                reco_base = (torch.matmul(X_p, reco_base.squeeze(1).T).T).reshape(reco_base.shape[0], 1, 96, 96)
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
            #for inum_, batch_val in enumerate(test_loader):
            inum_ = 0
            for batch_val in coeffs_saved_test:
                inum_ = inum_ + 1
                val_step += 1
                loss_C1_val = torch.FloatTensor([0.]).to(device)
                #batch_val = get_encoded_batch(batch_val,Q_exact)
                #batch_val = get_SmartGridBatch(batch_val,smart_indsX, smart_indsY)
                batch_val = batch_val.float()
                batch_val = batch_val.to('cuda')
                #print('training batch size: ', batch_val.shape)

                #batch_x[torch.where(batch_x < 0)] = 0
                #batch_x = batch_x / batch_x.max()


                #batch_val = batch_val.to(device)
                #print('val batch size: ', batch_val.shape)
                reconstruction_val = model_reg(batch_val)
                #reconstruction_val = reconstruction_val.view(batch_val.shape)
                image_batches_testp = image_batches_test[inum_-1]
                reconstructionH_val = (torch.matmul(X_p, reconstruction_val.squeeze(1).T).T).reshape(reconstruction_val.shape[0], 1, 96, 96)

                if reco_loss == 'mse':
                    loss_reconstruction_val = F.mse_loss(reconstructionH_val, image_batches_testp)
                    #print('reconstruction_val.shape',reconstruction_val.shape)

                    
                tmp_loss_list.append(float(loss_reconstruction_val.item()))

                if enable_wandb:
                    wandb.log({'rAE-loss_reco_val': float(loss_reconstruction_val.item())})
                    if (inum_ == 0):
                        '''im = torch.matmul(X_p, reconstruction_val[0])
                        im[torch.where(im<0)] = 0
                        im = im.reshape(96,96)'''
                        plt.imshow(reconstructionH_val[0][0].detach().cpu().numpy())
                        plt.title('Reg Reconstruction, step %d' % (epoch))
                        plt.colorbar()
                        wandb.log({"rAE-reco": plt})
                        plt.close()

                if enable_wandb:
                    wandb.log({'rAE-loss_reco_val2': float(loss_reconstruction_val.item())})
                    if (inum_ == 0):
                        '''im_or = torch.matmul(X_p, batch_val[0])
                        im_or[torch.where(im_or<0)] = 0
                        im_or = im_or.reshape(96,96)'''
                        plt.imshow(image_batches_testp[0][0].detach().cpu().numpy())
                        plt.title('Origional Image, step %d' % (epoch))
                        plt.colorbar()
                        wandb.log({"rAE-reco_2": plt})
                        plt.close()
                    
                if train_base_model:
                    reco_base = model(batch_val)#.view(batch_val.size())
                    #reconstruction_val = reconstruction_val.view(batch_val.shape)
                    #image_batches_test = image_batches_test[inum_-1]
                    reco_base = (torch.matmul(X_p, reco_base.squeeze(1).T).T).reshape(reco_base.shape[0], 1, 96, 96)
                    #loss_base_val_ = F.mse_loss(reco_base, batch_val)
                    loss_base_val_ = F.mse_loss(reco_base, image_batches_testp)
                    tmp_base_list.append(float(loss_base_val_.item()))
                    if enable_wandb:
                        wandb.log({'bAE-loss_reco_val': float(loss_base_val_.item())}) 
                        if (inum_ == 0):
                            '''im_b = torch.matmul(X_p, reco_base[0])
                            im_b[torch.where(im_b<0)] = 0
                            im_b = im_b.reshape(96,96)'''
                            plt.imshow(reco_base[0][0].detach().cpu().numpy())
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
        path = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/output/MRT_full/test_run_saving/'
        #path = './output/MRT_full/test_run_saving/'
        os.makedirs(path, exist_ok=True)
        name = '_'+reg_nodes_sampling+'_'+str(train_set_size)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)
        torch.save(loss_arr_reg, path+'/loss_arr_reg_RKMRI_TDA'+name)
        torch.save(loss_arr_reco, path+'/loss_arr_reco_RKMRI_TDA'+name)
        torch.save(loss_arr_base, path+'/loss_arr_base_RKMRI_TDA'+name)
        torch.save(loss_arr_val_reco, path+'/loss_arr_val_reco_RKMRI_TDA'+name)
        torch.save(loss_arr_val_base, path+'/loss_arr_val_base_RKMRI_TDA'+name)
        torch.save(model.state_dict(), path+'/model_base_RKMRI_TDA'+name)
        torch.save(model_reg.state_dict(), path+'/model_reg_RKMRI_TDA'+name)

        #torch.save(model, path+'/model_base_old_try'+name)
        #torch.save(model_reg, path+'/model_reg_old_try'+name)
        return model, model_reg, loss_arr_reg, loss_arr_reco, loss_arr_base, loss_arr_val_reco, loss_arr_val_base
    else:
        return model_reg, loss_arr_reg, loss_arr_reco