from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F


import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm 

import os
import re
import copy

import models as models
from AutoEncoder_exp import set_seed

from minterpy.tree import MultiIndicesTree
import wandb

import sys
sys.path.append('ext/pau')
from pau.utils import PAU

def runge(x, factor = 25):
    return 1 / (1+factor*np.linalg.norm(x**2, axis=1))


def train(degPoly, mlp_noLayers, mlp_noFeatures, activation_function = 'relu', runge_factor = 25, no_epochs = 2000, use_double = True, use_wandb = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.FloatTensor
    if(use_double):
        dtype = torch.DoubleTensor

    # validation points on equidistant grid
    x_1 = np.arange(-1,1,0.1).reshape((-1,1))
    x_2 = np.arange(-1,1,0.1).reshape((-1,1))
    xx, yy = np.meshgrid(x_1, x_2)
    x = np.concatenate((xx.reshape((-1,1)),yy.reshape((-1,1))), axis=1)

    if(use_wandb):
        # # Generate Training Data
        wandb.config.degPoly = degPoly
        wandb.config.dimension = 2
        wandb.config.mlp_noFeatures = mlp_noFeatures
        wandb.config.mlp_noLayers = mlp_noLayers
    

    ### Čebyšëv nodes
    set_seed(2342)
    interpol_tree = MultiIndicesTree(m=2, n=degPoly, lp_degree=2)
    pts_chebyshev = interpol_tree.grid_points.transpose()
    y_chebyshev = dtype(runge(pts_chebyshev, runge_factor)).to(device)
    pts_chebyshev = dtype(pts_chebyshev).to(device)
    noPts, _ = pts_chebyshev.shape

    ### RANDOM nodes
    pts_random = np.random.uniform(-1,1, [noPts,2])
    y_random = dtype(runge(pts_random, runge_factor)).to(device)
    pts_random = dtype(pts_random).to(device)
    
    ### EQUIDISTANT nodes
    noGridPts_ = int(np.sqrt(noPts))
    spacing = 2/noGridPts_
    x__1 = np.arange(-1,1,spacing).reshape((-1,1))
    x__2 = np.arange(-1,1,spacing).reshape((-1,1))
    xxx, yyy = np.meshgrid(x__1, x__2)
    pts_equidist = np.concatenate((xxx.reshape((-1,1)),yyy.reshape((-1,1))), axis=1)
    y_equidist = dtype(runge(pts_equidist, runge_factor)).to(device)
    pts_equidist = dtype(pts_equidist).to(device)

    # choose activation function
    if(activation_function == 'relu'):
        activation = torch.relu
    elif(activation_function == 'tanh'):
        activation = torch.tanh
    elif(activation_function == 'sigmoid'):
        activation = torch.sigmoid
    elif(activation_function == 'sin'):
        activation = torch.sin
    elif(activation_function == 'pau'):
        activation = PAU()
    
    # Train Networks
    set_seed(2342)
    myMLP_random = models.MLP(input_size=2, hidden_size=mlp_noFeatures, num_hidden=mlp_noLayers, output_size=1, activation = activation).to(device)
    myMLP_chebyshev = models.MLP(input_size=2, hidden_size=mlp_noFeatures, num_hidden=mlp_noLayers, output_size=1, activation = activation).to(device)
    myMLP_equidist = models.MLP(input_size=2, hidden_size=mlp_noFeatures, num_hidden=mlp_noLayers, output_size=1, activation = activation).to(device)
    
    myMLP_chebyshev = copy.deepcopy(myMLP_random)
    myMLP_equidist = copy.deepcopy(myMLP_random)
    
    if(use_double):
        myMLP_random = myMLP_random.double()
        myMLP_chebyshev = myMLP_chebyshev.double()
        myMLP_equidist = myMLP_equidist.double()

    # train on equidistant, random & chebyshev nodes   
    optimizer_random = torch.optim.LBFGS(myMLP_random.parameters(), lr=0.1, line_search_fn='strong_wolfe')
    optimizer_chebyshev = torch.optim.LBFGS(myMLP_chebyshev.parameters(), lr=0.1, line_search_fn='strong_wolfe')
    optimizer_equidist = torch.optim.LBFGS(myMLP_equidist.parameters(), lr=0.1, line_search_fn='strong_wolfe')
    
    for i in range(no_epochs):
        def closure():
            yihat_random = myMLP_random(pts_random).view(-1)
            loss = torch.mean((yihat_random - y_random)**2)
            optimizer_random.zero_grad()
            loss.backward()
            return loss

        def closure_cheby():
            optimizer_chebyshev.zero_grad()
            yihat_cheby = myMLP_chebyshev(pts_chebyshev).view(-1)
            loss_cheby = torch.mean((yihat_cheby - y_chebyshev)**2)
            loss_cheby.backward()
            return loss_cheby
        
        def closure_equidist():
            optimizer_equidist.zero_grad()
            yihat_equidist = myMLP_equidist(pts_equidist).view(-1)
            loss_equidist = torch.mean((yihat_equidist - y_equidist)**2)
            loss_equidist.backward()
            return loss_equidist
        
        optimizer_random.step(closure)
        optimizer_chebyshev.step(closure_cheby)
        optimizer_equidist.step(closure_equidist)

        if(use_wandb):
            wandb.log({'loss_random': closure().item()}, step=i)
            wandb.log({'loss_chebyshev': closure_cheby().item()}, step=i)
            wandb.log({'loss_equidist': closure_equidist().item()}, step=i)
        else:
            print("([%d] lr %.4e, lc %.4e, le %.4e" % (i, closure().item(), closure_cheby().item(), closure_equidist().item()))

    ### compute baseline performance
    y = runge(x, runge_factor)
    x = dtype(x).to(device)
    
    y_hat_random = myMLP_random(x).squeeze().detach().cpu().numpy()
    y_hat_equidist = myMLP_equidist(x).squeeze().detach().cpu().numpy()
    y_hat_chebyshev = myMLP_chebyshev(x).squeeze().detach().cpu().numpy()

    L_infty_random = np.max(abs(y_hat_random - y))
    L_infty_chebyshev = np.max(abs(y_hat_chebyshev - y))
    L_infty_equidist = np.max(abs(y_hat_equidist - y))

    if(use_wandb):
        wandb.log({'Linf_random': L_infty_random})
        wandb.log({'Linf_equidist': L_infty_equidist})
        wandb.log({'Linf_chebyshev': L_infty_chebyshev})
    else:
        print("Linf-r %.4e -- Linf-c %.4e -- Linf-e %.4e" % (L_infty_random, L_infty_chebyshev, L_infty_equidist))

    return myMLP_random, myMLP_chebyshev, myMLP_equidist

    
def main(): 
    set_seed(2342)
    
    os.environ['WANDB_API_KEY'] = 'd190f2bcc7b25562cc1e14760ba03fff5b7cf8ce'
    wandb.login()

    config_defaults = {
        'mlp_noFeatures': 50,
        'mlp_noLayers': 3,
        'degPoly': 5,
        'activationFunction': 'relu',
        'runge_factor': 25,
        'double': True
    }

    # Initialize a new wandb run
    wandb.init(config=config_defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    train(config.degPoly, config.mlp_noLayers, config.mlp_noFeatures, config.activationFunction, config.runge_factor, convfig.double)


if __name__ == '__main__':
    main()
