import torch
import numpy as np
from minterpy_in.utils import leja_ordered_values
#from minterpy.tree import MultiIndicesTree
from minterpy_in.utils import gamma_lp
#import vegas
from scipy.spatial import cKDTree
import itertools


def __gen_points_lp(m, N, Points, Gamma):
    PP = np.zeros((m, N))
    for i in range(N):
        for j in range(m):
            PP[j, i] = Points[j, int(Gamma[j, i])]
    return PP


def __chebyshevSampler(batch_size, latent_dim, n, lpDegree, samplingFactor = 100):
    #Gamma = np.random.randint(low=0, high=n+1, size=(latent_dim, samplingFactor*batch_size))
    
    # the probability to 0 will be lifted to 3n+1, while p(0 < n_i <= n) = 1/3n
    # @TODO: switch to Poisson distirbution to decrease rejection probability
    Gamma = np.random.randint(low=0, high=4*n, size=(latent_dim, samplingFactor*batch_size)) 
    Gamma[Gamma > n+1] = 0
    
    # filter elements that dont fulfill LP degree
    return Gamma[:, np.linalg.norm(Gamma, ord=lpDegree, axis=0) <= n]


def sampleChebyshevNodes__(batch_size, latent_dim, n, lpDegree = 2, samplingFactor = 100):
    ## iterative function to draw samples from Gamma
    #Gamma = np.random.randint(low=0, high=n+1, size=(latent_dim, samplingFactor*batch_size))
    
    ## filter elements that dont fulfill LP degree
    #Gamma = np.array([x for x in Gamma.transpose() if np.linalg.norm(x, ord = lpDegree) <= n]).transpose()
    Gamma = __chebyshevSampler(batch_size, latent_dim, n, lpDegree, samplingFactor)
    
    while(Gamma.shape[1] < batch_size):
        Gamma_sample = __chebyshevSampler(batch_size, latent_dim, n, lpDegree, samplingFactor)
        Gamma = np.concatenate((Gamma, Gamma_sample), axis = 1)
    
    Gamma = Gamma[:,0:batch_size]
    
    Points = np.zeros((latent_dim, n + 1))
    leja_values = leja_ordered_values(n)

    for i in range(latent_dim):
        Points[i] = (-1) ** (i + 1) * leja_values

    _, batch_size = Gamma.shape
        
    PP = __gen_points_lp(latent_dim, batch_size, Points, Gamma).transpose()

    return PP
    
    
    
def sampleChebyshevNodes(batch_size, latent_dim, weightJac, n):
    Gamma = np.random.randint(low=0, high=n, size=(latent_dim, batch_size))
    
    points = np.polynomial.chebyshev.chebgauss(n)[0][::-1]
    
    weights = np.polynomial.chebyshev.chebgauss(n)[1][::-1]
    
    PP = np.zeros((latent_dim, batch_size))
    for i in range(batch_size):
        for j in range(latent_dim):
            PP[j, i] = points[int(Gamma[j, i])]

    PP = PP.transpose()
            
    WW = np.ones((PP.shape[0]))
    if (weightJac):
        for inum, pts in enumerate(PP):
            weightprod = WW[inum]
            for pt in pts:
                weightprod *= weights[np.where(points==pt)[0]]
            WW[inum] = weightprod
            
    return PP, WW



def sampleLegendreNodes(batch_size, latent_dim, weightJac, points, weights, n):
    Gamma = np.random.randint(low=0, high=n, size=(latent_dim, batch_size))
    
    #points = np.polynomial.legendre.leggauss(n)[0][::-1]
    
    #weights = np.polynomial.legendre.leggauss(n)[1][::-1]
    
    PP = np.zeros((latent_dim, batch_size))
    for i in range(batch_size):
        for j in range(latent_dim):
            PP[j, i] = points[int(Gamma[j, i])]

    PP = PP.transpose()
            
    WW = np.ones((PP.shape[0]))
    if (weightJac):
        for inum, pts in enumerate(PP):
            weightprod = WW[inum]
            for pt in pts:
                weightprod *= weights[np.where(points==pt)[0]]
            WW[inum] = weightprod
            
    return PP, WW


def sampleNodes(nodes, szSample = 100):
    rand_nodes = torch.randperm(nodes.shape[0])[:szSample]
    return nodes[rand_nodes, :]


def computeC0Loss(nodes_t, nodes_t_cycle, idx_Element = 0):   
    loss_C0 = torch.mean( (nodes_t_cycle - nodes_t)**2 )
    return loss_C0


def computeC1Loss_deprecated(cheb_nodes, model, device):
    noNodes, szLatDim = cheb_nodes.shape
    I = torch.eye(szLatDim).unsqueeze(0).unsqueeze(2).to(device) # extract values of all minor diagonals (I = 1) 
    I_minDiag = -1 * torch.eye(szLatDim).unsqueeze(0).unsqueeze(2).to(device) + 1 # extract values of all minor diagonals (I = 1) while this matrix is zero on main diagonal
    f = lambda x: model.encoder(model.decoder(x.to(device))) # loop through autoencoder    
    Jac = torch.autograd.functional.jacobian(f, cheb_nodes.to(device), create_graph = True).squeeze() # compute Jacobian
    loss_C1 = torch.mean((Jac * I_minDiag)**2) + torch.mean(((Jac - Jac * I_minDiag) - I)**2) # extract + minimize values on minor diagonals
    return loss_C1, Jac

def computeC1Loss_(cheb_nodes, model, device, guidanceTerm = True):
    noNodes, szLatDim = cheb_nodes.shape
    I = torch.eye(szLatDim).unsqueeze(0).unsqueeze(2).to(device) # extract values of all minor diagonals (I = 1) 
    I_minDiag = -1 * torch.eye(szLatDim).unsqueeze(0).unsqueeze(2).to(device) + 1 # extract values of all minor diagonals (I = 1) while this matrix is zero on main diagonal
    f = lambda x: model.encoder(model.decoder(x.to(device))) # loop through autoencoder    
    Jac = torch.autograd.functional.jacobian(f, cheb_nodes.to(device), create_graph = True).squeeze() # compute Jacobian
    if guidanceTerm:
        loss_C1 = torch.mean((Jac * I_minDiag)**2) + torch.mean(((Jac - Jac * I_minDiag) - I)**2) # extract + minimize values on minor diagonals
    else: 
        loss_C1 = torch.mean((Jac * I_minDiag)**2)
    return loss_C1, Jac

def computeC1Loss(cheb_nodes, model, device, guidanceTerm = True):
    # extends C1 loss by guidance term
    noNodes, szLatDim = cheb_nodes.shape
    I = torch.eye(szLatDim).to(device) # extract values of all minor diagonals (I = 1) 
    f = lambda x: model.encoder(model.decoder(x.to(device))) # loop through autoencoder
    Jac = torch.autograd.functional.jacobian(f, cheb_nodes.to(device), create_graph = True).squeeze()
    loss_C1_arr = torch.zeros(noNodes).to(device)
    inum = 0
    for i in range(cheb_nodes.shape[0]):
    #for node_points in cheb_nodes:
        #node_points = torch.reshape(node_points, (1, szLatDim))
        #Jac = torch.autograd.functional.jacobian(f, node_points.to(device), create_graph = True).squeeze() # compute Jacobian
        loss_C1 = torch.mean((Jac[i,:,i,:] - I)**2)
        if(guidanceTerm):
            min_diag_val = torch.mean((torch.diagonal(Jac[i,:,i,:], dim1 = 0, dim2 = 1) - 1)**2)
            loss_C1 = loss_C1 + min_diag_val
        loss_C1_arr[inum] = loss_C1
        inum += 1
        
    #calculate jacobian of all the nodes together    
    Jac = torch.autograd.functional.jacobian(f, cheb_nodes.to(device), create_graph = True).squeeze() # compute Jacobian
    return torch.mean(loss_C1_arr), Jac


def computeC1Loss_upd(cheb_nodes, model, device, guidanceTerm = True):
    # extends C1 loss by guidance term
    noNodes, szLatDim = cheb_nodes.shape
    I = torch.eye(szLatDim).to(device) # extract values of all minor diagonals (I = 1) 
    f = lambda x: model.encoder(model.decoder(x.to(device))) # loop through autoencoder
    #print('I.shape', I.shape)

    loss_C1_arr = torch.zeros(noNodes).to(device)
    #cheb_nodes = cheb_nodes.unsqueeze(1)
    
    #Jac = torch.autograd.functional.jacobian(f, cheb_nodes.to(device), create_graph = True).squeeze()
    #print('Jac.shape', Jac.shape)
    Jac_array = torch.tensor([]).to(device)
    inum = 0
    for node_points in cheb_nodes:
        node_points = torch.reshape(node_points, (1, szLatDim))
        #print('node_points.shape', node_points.shape)
        Jac = torch.autograd.functional.jacobian(f, node_points.to(device), create_graph = True).squeeze() # compute Jacobian
        #print('Jac.shape', Jac.shape) 
        loss_C1 = torch.mean((Jac - I)**2)
        if(guidanceTerm):
            min_diag_val = torch.mean((torch.diagonal(Jac, dim1 = 0, dim2 = 1) - 1)**2)
            loss_C1 = loss_C1 + min_diag_val
        loss_C1_arr[inum] = loss_C1
        Jac_array = torch.cat((Jac_array, Jac.unsqueeze(0)))
        inum += 1
        
    #calculate jacobian of all the nodes together    
    Jac_array = Jac_array.unsqueeze(1)
    #print('Jac_array.shape', Jac_array.shape)
    #print('loss_C1_arr.shape', loss_C1_arr.shape)
    #Jac = torch.autograd.functional.jacobian(f, cheb_nodes.to(device), create_graph = True).squeeze() # compute Jacobian
    return torch.mean(loss_C1_arr), Jac_array

def computeC1LossWeighted_(cheb_nodes, weights_subsample, model, device, guidanceTerm = True):
    # extends C1 loss by guidance term
    noNodes, szLatDim = cheb_nodes.shape
    I = torch.eye(szLatDim).to(device) # extract values of all minor diagonals (I = 1) 
    f = lambda x: model.encoder(model.decoder(x.to(device))) # loop through autoencoder
    
    loss_C1_arr = torch.zeros(noNodes).to(device)
    inum = 0
    for node_points, node_weight in zip(cheb_nodes, weights_subsample):
        node_points = torch.reshape(node_points, (1, szLatDim))
        Jac = torch.autograd.functional.jacobian(f, node_points.to(device), create_graph = True).squeeze() # compute Jacobian
        loss_C1 = node_weight * torch.sum((Jac - I)**2) #weighted sum
        if(guidanceTerm):
            min_diag_val = node_weight * torch.sum((torch.diagonal(Jac, dim1 = 0, dim2 = 1) - 1)**2)
            loss_C1 = loss_C1 + min_diag_val
        loss_C1_arr[inum] = loss_C1
        inum += 1   
    Jac = torch.autograd.functional.jacobian(f, cheb_nodes.to(device), create_graph = True).squeeze() # compute Jacobian
    return torch.sum(loss_C1_arr), Jac

def computeC1LossWeighted(cheb_nodes, weights_subsample, model, device, guidanceTerm = True):
    # extends C1 loss by guidance term
    noNodes, szLatDim = cheb_nodes.shape
    I = torch.eye(szLatDim).to(device) # extract values of all minor diagonals (I = 1) 
    f = lambda x: model.encoder(model.decoder(x.to(device))) # loop through autoencoder
    #print('jac ', cheb_nodes.shape)
    Jac = torch.autograd.functional.jacobian(f, cheb_nodes.to(device), create_graph = True).squeeze()
    loss_C1_arr = torch.zeros(noNodes).to(device)
    inum = 0
    for i in range(weights_subsample.shape[0]):
        loss_C1 = weights_subsample[i] * torch.sum((Jac[i,:,i,:] - I)**2) #weighted sum
        if(guidanceTerm):
            min_diag_val = weights_subsample[i] * torch.sum((torch.diagonal(Jac[i,:,i,:], dim1 = 0, dim2 = 1) - 1)**2)
            loss_C1 = loss_C1 + min_diag_val
        loss_C1_arr[inum] = loss_C1
        inum += 1   
    #Jac = torch.autograd.functional.jacobian(f, cheb_nodes.to(device), create_graph = True).squeeze() # compute Jacobian
    return torch.sum(loss_C1_arr), Jac


def computeC1LossWeighted_without_loop(cheb_nodes, weights_subsample, model, device, guidanceTerm = True):
    noNodes, szLatDim = cheb_nodes.shape
    I = torch.eye(szLatDim).to(device) # extract values of all minor diagonals (I = 1) 
    f = lambda x: model.encoder(model.decoder(x.to(device))) # loop through autoencoder

    Jac = torch.autograd.functional.jacobian(f, cheb_nodes.to(device), create_graph = True).squeeze()
    weights_subsample_ = torch.repeat_interleave(weights_subsample, repeats=szLatDim, dim=0)
    ind = [x+noNodes*(szLatDim-1)*i+(noNodes+1)*i for i in range(0,noNodes) for x in range(0,szLatDim*noNodes,noNodes)]
    loss_C1=torch.sum((Jac.reshape(noNodes**2*szLatDim,szLatDim)[ind]
                       - torch.cat(noNodes*[I]))**2 * weights_subsample_[:, None].to(device))
    
    if guidanceTerm:
        min_diag_val = torch.sum((
             weights_subsample.unsqueeze(1).to(device) *
            (torch.diagonal(Jac.reshape(noNodes**2*szLatDim,szLatDim)[ind].reshape(noNodes, szLatDim,szLatDim),
                            dim1=1, dim2=2) - 1)**2))
        loss_C1 = loss_C1 + min_diag_val
    return loss_C1, Jac

def computeC2LossOnSingleChebNode(cheb_nodes, model, noChebNode = 0, dim = 1):
    noNodes, szLatDim = cheb_nodes.shape
    cheb_node = cheb_nodes[noChebNode:noChebNode+1, :]
    f = lambda x: model.encoder(model.decoder(x))[:,dim] # loop through autoencoder    
    Hes = torch.autograd.functional.hessian(f, cheb_node, create_graph = True).squeeze() # compute Hessian
    loss_C2 = torch.mean(Hes**2) # Hessian is supposed to vanish
    return loss_C2, Hes



def FindC1Loss(node_points, I, f, szLatDim, device, guidanceTerm):
    node_points = torch.reshape(node_points, (1, szLatDim))
    Jac = torch.autograd.functional.jacobian(f, node_points.to(device), create_graph = False).squeeze() # compute Jacobian
    loss_C1 = torch.sum((Jac - I)**2)
    if(guidanceTerm):
        min_diag_val = torch.sum((torch.diagonal(Jac, dim1 = 0, dim2 = 1) - 1)**2)
        loss_C1 = loss_C1 + min_diag_val
    return loss_C1


'''
def adapt_vegas(initial_points, function_val, limits, grid_to_map=None, vegas_iter=10, vegas_alpha=0.7):
    function_val = function_val.detach().numpy()
    initial_points = np.array(initial_points, dtype=np.float64)
    admap = vegas.AdaptiveMap(limits, ninc=5)
    x = np.zeros(initial_points.shape, float)            # work space
    jac = np.zeros(initial_points.shape[0], float)
 
    #map to new grid
    admap.adapt_to_samples(initial_points, function_val, nitn=vegas_iter, alpha=vegas_alpha)
    admap.map(initial_points, x, jac)
    admap.clear()
 
    if(grid_to_map is not None):
        ##tree for finding the nearest point
        tree = cKDTree(grid_to_map)
 
        #find the closed points in the grid correspoinding to the newly mapped points
        mod_x = []
        for inum,pt in enumerate(x):
            dd, ii = tree.query(pt)
            temp_x = grid_to_map[ii]
            jj = np.where(np.all(initial_points==temp_x,axis=1))[0]
            if (jj.size==0):
                mod_x.append(temp_x)
        x = np.array(mod_x, dtype=np.float64)
        x = np.unique(x, axis=0)
  
    return x, jac


def vegasC1Loss(szSample, latent_dim, model, device, limits, guidanceTerm = True):
    I = torch.eye(latent_dim).to(device) # extract values of all minor diagonals (I = 1) 
    f = lambda x: model.encoder(model.decoder(x.to(device))) # loop through autoencoder
    loss_C1_arr = torch.zeros(szSample).to(device)
    sample = torch.FloatTensor(np.random.uniform(low=-1, high=1, size=(szSample,latent_dim))).to(device)
    for inum, node_points in enumerate(sample):
        loss_C1_arr[inum] = FindC1Loss(node_points, I, f, latent_dim, device, guidanceTerm)
        
    newpoints, jac_vegas = adapt_vegas(sample.cpu(), loss_C1_arr.cpu(), limits, grid_to_map=None, vegas_iter=5, vegas_alpha=0.7)
    
    loss_C1_cum = torch.zeros(1).to(device)
    
    cheb_nodes = torch.FloatTensor(newpoints).to(device)
    weights_subsample = torch.FloatTensor(jac_vegas).to(device)
    
    for node_points, node_weight in zip(cheb_nodes, weights_subsample):
        node_points = torch.reshape(node_points, (1, latent_dim))
        Jac = torch.autograd.functional.jacobian(f, node_points.to(device), create_graph = True).squeeze() # compute Jacobian
        loss_C1 = node_weight * torch.sum((Jac - I)**2) #weighted sum
        if(guidanceTerm):
            min_diag_val = node_weight * torch.sum((torch.diagonal(Jac, dim1 = 0, dim2 = 1) - 1)**2)
            loss_C1 = loss_C1 + min_diag_val
        loss_C1_cum += loss_C1
    
    return loss_C1_cum/(cheb_nodes.shape[0])
'''
def quadC1Loss(qpoints, qweights, model, device, guidanceTerm = True):
    # extends C1 loss by guidance term
    noNodes, szLatDim = qpoints.shape
    I = torch.eye(szLatDim).to(device) # extract values of all minor diagonals (I = 1) 
    f = lambda x: model.encoder(model.decoder(x.to(device))) # loop through autoencoder
    
    loss_C1_arr = torch.zeros(noNodes).to(device)
    inum = 0
    for node_points, node_weight in zip(qpoints, qweights):
        node_points = torch.reshape(node_points, (1, szLatDim))
        Jac = torch.autograd.functional.jacobian(f, node_points.to(device), create_graph = True).squeeze() # compute Jacobian
        loss_C1 = node_weight * torch.sum((Jac - I)**2) #weighted sum
        if(guidanceTerm):
            min_diag_val = node_weight * torch.sum((torch.diagonal(Jac, dim1 = 0, dim2 = 1) - 1)**2)
            loss_C1 = loss_C1 + min_diag_val
        loss_C1_arr[inum] = loss_C1
        inum += 1
        
    return torch.sum(loss_C1_arr)
