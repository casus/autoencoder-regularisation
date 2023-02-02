import sys
sys.path.append('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders')

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

def train(train_loader, test_loader, no_channels, dx, dy, no_epochs=2, TDA=0.4, reco_loss='mse', latent_dim=10, 
          hidden_size=1024, no_layers=3, activation = F.relu, lr = 3e-4, alpha=1., bl=False, 
          seed = 2342, train_base_model=False, no_samples=5, deg_poly=10,
          reg_nodes_sampling="legendre_exp", no_val_samples = 10, use_guidance = True,
          enable_wandb=True, wandb_project=None, wandb_entity=None):


    wass_outputs = []
    wass_outputs_val = []

    points = np.polynomial.legendre.leggauss(deg_poly)[0][::-1]
    
    weights = np.polynomial.legendre.leggauss(deg_poly)[1][::-1]


    weight_jac = False
    if enable_wandb:
        import wandb
        wandb.init(project='Test_mrt', entity='ae_reg_team')
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
    
    #train_loader, test_loader, no_channels, dx, dy = getDataset('FashionMNIST', 200, False)

    set_seed(2342)

    #inp_dim = [no_channels, dx, dy]

    inp_dim = 1024

    '''linear_size = dx
    xvals = np.linspace(-1.0, 1.0, linear_size)
    X,Y = np.meshgrid(xvals, xvals)
    coords = np.zeros((linear_size**2,2))
    coords[:,0] = X.reshape(-1)
    coords[:,1] = Y.reshape(-1)
    sample_points = coords

    multi_index = mp.MultiIndexSet.from_degree(spatial_dimension = 2, poly_degree = 20, lp_degree = 2.0)
    grid_obj = mp.Grid(multi_index)
    regressor = mp.Regression(grid_obj)
    rand_func = np.random.rand((1024))
    regressor.regression(sample_points, rand_func)
    R_matrix = torch.FloatTensor(regressor.regression_matrix).to(device)
    print('R_matrix', R_matrix.shape)'''

    '''def get_lagrange_coeffs(batch_x):
        batch_coeffs = torch.tensor([])
        for i in range(batch_x.shape[0]):
            function_vals = batch_x[i][0].reshape(batch_x[0].squeeze(0).shape[0]*batch_x[0].squeeze(0).shape[1]).cpu().detach().numpy()
            regressor.regression(sample_points, function_vals)
            lag_poly_coeffs_ = regressor._lagrange_poly.coeffs
            lag_poly_coeffs = torch.tensor(lag_poly_coeffs_)
            batch_coeffs = torch.cat((batch_coeffs, lag_poly_coeffs),0)
            #print(lag_poly_coeffs_)
        batch_coeffs = batch_coeffs.reshape(batch_x.shape[0], lag_poly_coeffs.shape[0])
        batch_coeffs = (batch_coeffs-batch_coeffs.min()) / (batch_coeffs.max() - batch_coeffs.min())
        return batch_coeffs'''

    model_reg = AE(inp_dim, hidden_size, latent_dim, 
                       no_layers, activation).to(device) # regularised autoencoder

    if train_base_model:
        model = AE(inp_dim, hidden_size, latent_dim, 
                       no_layers, activation).to(device) # baseline autoencoder
        model = copy.deepcopy(model_reg)

    #print('model_reg', model_reg)
    #print('model', model)                                                                                       
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

    #generating points on 15 dim-circle

    Jac_val_pts = torch.FloatTensor(np.random.uniform(-1,1,size=(no_val_samples, latent_dim))).to(device)
    


    batch_size_cfs = 200
    #coeffs_saved_trn = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/AllFmnistTrainRKCoeffsDeg25.pt').to(device)
    image_batches_trn = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/torus_dataset/1024_dim_torus_24000pts.pt').to(device)
    #print("what is this ? ", image_batches_trn.shape)
    #print(int(image_batches_trn.shape[0]/batch_size_cfs))

    image_batches_trn = image_batches_trn.reshape(int(image_batches_trn.shape[0]/batch_size_cfs), batch_size_cfs, 1, inp_dim)
    #coeffs_saved_trn = coeffs_saved_trn.reshape(int(coeffs_saved_trn.shape[0]/batch_size_cfs), batch_size_cfs, coeffs_saved_trn.shape[1]).unsqueeze(2) 


    #coeffs_saved_test = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/AllFmnistTestRKCoeffsDeg25.pt').to(device)
    image_batches_test = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/torus_dataset/1024_dim_torus_24000pts.pt').to(device)
    image_batches_test = image_batches_test[:400]
    image_batches_test = image_batches_test.reshape(int(image_batches_test.shape[0]/batch_size_cfs), batch_size_cfs, 1, inp_dim)
    #coeffs_saved_test = coeffs_saved_test[:11200]
    #coeffs_saved_test = coeffs_saved_test.reshape(int(coeffs_saved_test.shape[0]/batch_size_cfs), batch_size_cfs, coeffs_saved_test.shape[1]).unsqueeze(2) 


    #print('coeffs_saved_trn.shape',coeffs_saved_trn.shape)



    #print('coeffs_saved_test.shape',coeffs_saved_test.shape)
    image_batches_trn = image_batches_trn[:int(image_batches_trn.shape[0]*TDA)]
    image_batches_test = image_batches_test[:int(image_batches_test.shape[0]*TDA)]

    print('image_batches_trn.shape',image_batches_trn.shape)
    print('image_batches_test.shape',image_batches_test.shape)


    for epoch in tqdm(range(no_epochs)):
        
        loss_full = []
        loss_rec = []
        loss_rec_base = []
        loss_c1 = []
        print('Epoch : '+str(epoch)+ 'started')
        inum = 0
        #for inum, batch_x in enumerate(train_loader):
        #inum = 0
        for batch_x in image_batches_trn:    
            inum = inum+1
            #print('batch_x.shape before 0', len(batch_x))
            #batch_x = batch_x[0]    
            #print('batch_x.shape after 0', batch_x.shape)

            #batch_x = get_lagrange_coeffs(batch_x)
            batch_x = batch_x.float()
            global_step += 1
            loss_C1 = torch.FloatTensor([0.]).to(device) 
            # plain reconstruction using AE
            batch_x = batch_x.to(device)
            #print('training batch size: ', batch_x.shape)
            #print('batch_x.max()',batch_x.max())
            #print('batch_x.min()',batch_x.min())
            reconstruction = model_reg(batch_x)
            reconstruction = reconstruction.view(batch_x.shape)
            #batch_xH = torch.matmul(R_matrix, batch_x.squeeze(1).T).T
            #reconstructionH = torch.matmul(R_matrix, reconstruction.squeeze(1).T).T
            
            #print('batch_xH.shape',batch_xH.shape)
            #print('reconstructionH.shape',reconstructionH.shape)

            if reco_loss == 'mse':
                loss_reconstruction = F.mse_loss(reconstruction, batch_x)
            if reco_loss == 'wasserstein':
                '''n_mar = batch_x.shape[2]
                n_t_mar = batch_x.shape[3]
                wass_outputs = []
                for i in range(reconstruction.shape[0]):
                    with torch.no_grad():
                        batch_sum = torch.sum(batch_x[i][0])
                        recon_sum = torch.sum(reconstruction[i][0])
                    #ori_im = torch.unsqueeze(batch_x[i], 0) / batch_sum
                    #recon = torch.unsqueeze(reconstruction[i], 0) / recon_sum
                    ori_im = batch_x[i][0] / batch_sum
                    recon = reconstruction[i][0] / recon_sum
                    #wass_dist, _, _ = sinkhorn(recon, ori_im)
                    wass_dist = spc.sinkhorn_loss(ori_im,recon,epsilon, n_mar, niter)
                    wass_dist = torch.unsqueeze(wass_dist, 0) 
                    wass_outputs.append(wass_dist)
                result_wass = torch.cat(wass_outputs, dim=0)  #shape (64, 32*in_channels, 224, 224)
                #loss_reconstruction = torch.sqrt(torch.mean(torch.square(result_wass)))
                #print("result_wass",result_wass)'''
                loss_reconstruction = swd(batch_x, reconstruction, device="cuda")
                #loss_reconstruction = torch.sum(result_wass)



            if(reg_nodes_sampling == 'chebyshev'):
                nodes_subsample_np, weights_subsample_np = sampleChebyshevNodes(no_samples, latent_dim, weight_jac, n=deg_poly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)
                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
            elif(reg_nodes_sampling == 'legendre'): 
                nodes_subsample_np, weights_subsample_np = sampleLegendreNodes(no_samples, latent_dim, weight_jac, points, weights, n=deg_poly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)
                #print('nodes_subsample.shape', nodes_subsample.shape)
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
            for batch_val in image_batches_test:
                inum_ = inum_ + 1
                batch_val = batch_val[0]
                #batch_val = get_lagrange_coeffs(batch_val)
                batch_val = batch_val.float()
                val_step += 1
                loss_C1_val = torch.FloatTensor([0.]).to(device)
                batch_val = batch_val.to(device)
                #print('val batch size: ', batch_val.shape)
                reconstruction_val = model_reg(batch_val)
                reconstruction_val = reconstruction_val.view(batch_val.shape)
                #batch_valH = torch.matmul(R_matrix, batch_val.squeeze(1).T).T
                #reconstruction_valH = torch.matmul(R_matrix, reconstruction_val.squeeze(1).T).T
                #print('reconstruction_val.shape',reconstruction_val.shape)
                #print('batch_val.shape',batch_val.shape)

                if reco_loss == 'mse':
                    loss_reconstruction_val = F.mse_loss(reconstruction_val, batch_val)
                if reco_loss == 'wasserstein':
                    '''for i in range(reconstruction_val.shape[0]):
                        ori_im_val = batch_val[i][0] / torch.sum(batch_val[i][0])
                        recon_val = reconstruction_val[i][0] / torch.sum(reconstruction_val[i][0])
                        #wass_dist_val, _, _ = sinkhorn(recon_val, ori_im_val)
                        wass_dist_val = spc.sinkhorn_loss(ori_im_val,recon_val,epsilon,n_mar, niter)
                        wass_dist_val = torch.unsqueeze(wass_dist_val, 0) 
                        wass_outputs_val.append(wass_dist_val)
                    result_wass_val = torch.cat(wass_outputs_val, dim=0)  #shape (64, 32*in_channels, 224, 224)
                    #loss_reconstruction_val = torch.sqrt(torch.mean(torch.square(result_wass_val)))
                    loss_reconstruction_val = torch.sum(result_wass_val)'''
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

                        '''lag_poly_new = mp.LagrangePolynomial(multi_index, reconstruction_val[kk][0].cpu().detach().numpy())
                        l2n =  mp.get_transformation(lag_poly_new, mp.NewtonPolynomial)
                        newt_poly = l2n()
                        plt.imshow(newt_poly(sample_points).reshape(32,32))
                        plt.title('Wasserstein Reconstruction, step %d' % (epoch))
                        plt.colorbar()'''

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


                            '''lag_poly_new = mp.LagrangePolynomial(multi_index, reco_base[kk][0].cpu().detach().numpy())
                            l2n =  mp.get_transformation(lag_poly_new, mp.NewtonPolynomial)
                            newt_poly = l2n()
                            plt.imshow(newt_poly(sample_points).reshape(32,32))
                            plt.title('Wasserstein Reconstruction, step %d' % (epoch))
                            plt.colorbar()'''


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
        torch.save(loss_arr_reg, path+'/loss_arr_reg_1024tor24000'+name)
        torch.save(loss_arr_reco, path+'/loss_arr_reco_1024tor24000'+name)
        torch.save(loss_arr_base, path+'/loss_arr_base_1024tor24000'+name)
        torch.save(loss_arr_val_reco, path+'/loss_arr_val_reco_1024tor24000'+name)
        torch.save(loss_arr_val_base, path+'/loss_arr_val_base_1024tor24000'+name)
        torch.save(model.state_dict(), path+'/model_base_1024tor24000'+name)
        torch.save(model_reg.state_dict(), path+'/model_reg_1024tor24000'+name)

        #torch.save(model, path+'/model_base_old_try'+name)
        #torch.save(model_reg, path+'/model_reg_old_try'+name)        
        return model, model_reg, loss_arr_reg, loss_arr_reco, loss_arr_base, loss_arr_val_reco, loss_arr_val_base
    else:
        return model_reg, loss_arr_reg, loss_arr_reco