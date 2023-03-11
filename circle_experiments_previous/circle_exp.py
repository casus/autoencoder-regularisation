import sys
sys.path.append('/home/ramana44/autoencoder-regularisation-')

import math
import numpy as np
from models import AE
import matplotlib.pyplot as plt
import torch.nn as nn
from activations import Sin
import copy
from  models_circle import ConvAE
from models_circle import BetaVAE, reconstruction_loss, kl_divergence, MLPVAE

pi = math.pi

def PointsInCircum(r,n=100):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]

def PointsInCircumNDim(points, transform_to_nD):
    circle_nD = np.matmul(points, transform_to_nD)
    return circle_nD

##################################################################################################################

##################################################################################################################

##################################################################################################################
import torch
I = torch.eye(5)
print(I)
##################################################################################################################

n = 1000
points = PointsInCircum(1.,1000)
arr_points = np.array(points)
plt.scatter(arr_points[:,0], arr_points[:,1])
plt.grid(True)
plt.show()
plt.savefig('/home/ramana44/autoencoder-regularisation-/circle_experiments_previous/imagesaves/orig_circle.png')
plt.close()
##################################################################################################################
transform_to_3D = np.random.rand(2, 3)
print(transform_to_3D)
circle_3D = np.matmul(arr_points, transform_to_3D, )
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(circle_3D[:,0], circle_3D[:,1], circle_3D[:,2])

plt.show()
plt.savefig('/home/ramana44/autoencoder-regularisation-/circle_experiments_previous/imagesaves/circleIn3DSpace.png')
plt.close()
##################################################################################################################
import torch
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return self.data[:, index]

def get_loader(data):
    dataset = Dataset(data)
    sampler = torch.utils.data.SubsetRandomSampler(list(range(data.shape[0])))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)
    return loader

dim = 15
transform_to_nD = 4*np.random.rand(2, dim)-2
print(transform_to_nD)

data_tr = torch.from_numpy(PointsInCircumNDim(PointsInCircum(1.,3), transform_to_nD)).float()
data_val = torch.from_numpy(PointsInCircumNDim(PointsInCircum(1.,50), transform_to_nD)).float()
loader_tr = get_loader(data_tr)
loader_val = get_loader(data_val)
##################################################################################################################
model = AE(dim, 6, 2, 2, Sin()).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
##################################################################################################################
model_conv = ConvAE(dim, no_filters=5, no_layers=2, 
                    kernel_size=3, latent_dim=2, activation=Sin()).to(device)
pytorch_total_params = sum(p.numel() for p in model_conv.parameters() if p.requires_grad)
print(pytorch_total_params)
#print(model_conv)
##################################################################################################################
#del model_reg
hidden_size = 6
no_layers = 2
lr = 5e-3

no_filters = 5
kernel_size = 3
no_layers_conv = 2

'''
del model
del model_reg
del model_reg_ran
del model_reg_cheb
del model_reg_leg
'''

model = AE(dim, hidden_size, 2, no_layers, Sin()).to(device)
model_reg_tr = AE(dim, hidden_size, 2, no_layers, Sin()).to(device)
model_reg_ran = AE(dim, hidden_size, 2, no_layers, Sin()).to(device)
model_reg_cheb = AE(dim, hidden_size, 2, no_layers, Sin()).to(device)
model_reg_leg = AE(dim, hidden_size, 2, no_layers, Sin()).to(device)
model_conv = ConvAE(dim, no_filters, no_layers_conv, 
                    kernel_size, latent_dim=2, activation=Sin()).to(device)
model_vae = BetaVAE(dim, no_filters, no_layers_conv,
                  kernel_size, latent_dim=2, activation = Sin(), beta=0.01, use_mu=1.).to(device)
model_mlp_vae = MLPVAE(dim, hidden_size, 2, no_layers, beta=0.01).to(device)

#model_reg_tr = copy.deepcopy(model)
#model_reg_ran = copy.deepcopy(model)
#model_reg_cheb = copy.deepcopy(model)
#model_reg_leg = copy.deepcopy(model)

no_epochs = 550
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_tr = torch.optim.Adam(model_reg_tr.parameters(), lr=lr)
optimizer_ran = torch.optim.Adam(model_reg_ran.parameters(), lr=lr)
optimizer_cheb = torch.optim.Adam(model_reg_cheb.parameters(), lr=lr)
optimizer_leg = torch.optim.Adam(model_reg_leg.parameters(), lr=lr)
optimizer_mlp_vae = torch.optim.Adam(model_mlp_vae.parameters(), lr=1e-3)

optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=5e-3)
optimizer_vae = torch.optim.Adam(model_vae.parameters(), lr=5e-3)

mod_loss = []
mod_loss_tr = []
mod_loss_ran = []
mod_loss_cheb = []
mod_loss_leg = []
mod_loss_conv = []
mod_loss_vae = []
mod_loss_mlp_vae = []

for epoch in range(no_epochs):
    optimizer.zero_grad()
    model_output = model(data_tr.to(device))
    loss = torch.nn.MSELoss()(model_output, data_tr.to(device))
    mod_loss.append(float(loss.item()))
    loss.backward()
    optimizer.step()
    
plt.plot(list(range(0,no_epochs)), mod_loss, label='baseline, '+str(mod_loss[-1]))
plt.xlabel("$epoch$")
plt.ylabel("$loss$")
plt.title("Baseline model")
plt.grid(True)
plt.show()
plt.savefig('/home/ramana44/autoencoder-regularisation-/circle_experiments_previous/imagesaves/baselineLoss.png')
plt.close()
from regularisers_without_vegas import computeC1Loss, sampleChebyshevNodes, sampleLegendreNodes

'''regNodesSamplings = (["trainingData", "random", "chebyshev", "legendre",
                    "conv", "vae", "mlp_vae"])'''
                    
regNodesSamplings = (["trainingData", "random", "chebyshev", "legendre",
                    "conv"])

'''models = ([model_reg_tr, model_reg_ran, model_reg_cheb, model_reg_leg,
        model_conv, model_vae, model_mlp_vae])'''

models = ([model_reg_tr, model_reg_ran, model_reg_cheb, model_reg_leg,
        model_conv])

optimizers = ([optimizer_tr, optimizer_ran, optimizer_cheb, optimizer_leg, 
            optimizer_conv, optimizer_vae, optimizer_mlp_vae])
szSample = 10
latent_dim = 2
weightJac = False
degPoly=20
alpha = 0.1
for ind, model_reg in enumerate(models):
    mod_loss_reg = []
    regNodesSampling = regNodesSamplings[ind]
    print(regNodesSampling)
    optimizer = optimizers[ind]
    #print(mod_loss_reg)
    for epoch in range(no_epochs):
        if (regNodesSampling != "conv") and (regNodesSampling != "vae") and (regNodesSampling != "mlp_vae"):
            model_output = model_reg(data_tr.to(device))
            loss = torch.nn.MSELoss()(model_output, data_tr.to(device))
            mod_loss_reg.append(float(loss.item()))

            if(regNodesSampling == 'chebyshev'):
                nodes_subsample_np, weights_subsample_np = sampleChebyshevNodes(szSample, latent_dim, weightJac, n=degPoly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)
                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
            elif(regNodesSampling == 'legendre'): 
                nodes_subsample_np, weights_subsample_np = sampleLegendreNodes(szSample, latent_dim, weightJac, n=degPoly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)

                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
            elif(regNodesSampling == 'random'):
                nodes_subsample = torch.FloatTensor(szSample, latent_dim).uniform_(-1, 1)
            elif(regNodesSampling == 'trainingData'):
                nodes_subsample = model_reg.encoder(data_tr[0:szSample, :].to(device))

            loss_C1, Jac = computeC1Loss(nodes_subsample, model_reg, device, guidanceTerm = False) #

            loss = (1.-alpha)*loss + alpha*loss_C1
        if regNodesSampling == "conv":
            model_output = model_reg(data_tr.to(device)).squeeze(1)
            loss = torch.nn.MSELoss()(model_output, data_tr.to(device))
            mod_loss_reg.append(float(loss.item()))
        if regNodesSampling == "vae":
            x_reco, mu, logvar = model_reg(data_tr.to(device))
            recon_loss = torch.nn.MSELoss()(data_tr.to(device), x_reco.squeeze(1))
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss = recon_loss + model_vae.beta*total_kld
            mod_loss_reg.append(float(recon_loss.item()))
        if regNodesSampling == "mlp_vae":
            x_reco, mu, logvar = model_reg.forward(data_tr.to(device))
            recon_loss = torch.nn.MSELoss()(data_tr.to(device), x_reco)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss = recon_loss + model_vae.beta*total_kld
            mod_loss_reg.append(float(recon_loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(list(range(0,no_epochs)), mod_loss_reg, label=regNodesSampling+', '+str(mod_loss_reg[-1]))
    plt.xlabel("$epoch$")
    plt.ylabel("$loss$")
    plt.legend()
    plt.grid(True)
plt.show()
plt.savefig('/home/ramana44/autoencoder-regularisation-/circle_experiments_previous/imagesaves/allLosses.png')
plt.close()
for opt in optimizers:
    del opt
##################################################################################################################

##################################################################################################################
import matplotlib.pyplot as plt
points_tr = (model.encoder(data_tr.to(device))).detach().cpu().numpy()
points_val = (model.encoder(data_val.to(device))).detach().cpu().numpy()
plt.scatter(points_val[:,0], points_val[:,1], color="orange")
plt.scatter(points_tr[:,0], points_tr[:,1], color="blue")

plt.scatter(arr_points[:,0], arr_points[:,1], color='grey', alpha=0.05)
plt.grid(True)
#plt.title("Baseline: training data")
plt.show()


#labels = ["Reg on training data", "Reg on random points", "Reg on chebyshev nodes", "Reg on legendre nodes","conv", "vae","mlp_vae"]

labels = ["Reg on training data", "Reg on random points", "Reg on chebyshev nodes", "Reg on legendre nodes","conv"]

for ind, model_reg in enumerate(models):
    if ind < 5:
        points_tr = (model_reg.encoder(data_tr.to(device))).detach().cpu().numpy()
        points_val = (model_reg.encoder(data_val.to(device))).detach().cpu().numpy()
        plt.scatter(points_val[:,0], points_val[:,1], label='validation samples', color="orange")
        plt.scatter(points_tr[:,0], points_tr[:,1], label='training samples', color="blue")
        plt.scatter(arr_points[:,0], arr_points[:,1], color='grey', alpha=0.05)
        plt.grid(True)
        plt.title(labels[ind])
        plt.show()
        plt.savefig('/home/ramana44/autoencoder-regularisation-/circle_experiments_previous/imagesaves/s'+str(ind)+'.png')
        plt.close()
    else:
        points_tr, _, _ = (model_reg.encode(data_tr.to(device), False))
        points_tr = points_tr.detach().cpu().numpy()
        points_val,_ ,_ = (model_reg.encode(data_val.to(device), False))
        points_val = points_val.detach().cpu().numpy()
        plt.scatter(points_val[:,0], points_val[:,1], label='validation samples', color="orange")
        plt.scatter(points_tr[:,0], points_tr[:,1], label='training samples', color="blue")
        plt.scatter(arr_points[:,0], arr_points[:,1], color='grey', alpha=0.05)
        plt.grid(True)
        plt.title(labels[ind])
        plt.show()
        plt.savefig('/home/ramana44/autoencoder-regularisation-/circle_experiments_previous/imagesaves/s'+str(ind)+'.png')
        plt.close()
    del model_reg
##################################################################################################################


##################################################################################################################


##################################################################################################################


##################################################################################################################


##################################################################################################################


##################################################################################################################


##################################################################################################################


##################################################################################################################
