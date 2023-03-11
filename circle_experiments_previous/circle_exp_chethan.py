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

from torchvision import datasets, transforms

transform = transforms.ToTensor()

from models_for_circle import ConvoAE, Autoencoder_linear, VAE_mlp_circle_new
from loss_functions import contractive_loss_function, loss_fn_mlp_vae
from torch.autograd import Variable

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

data_tr = torch.from_numpy(PointsInCircumNDim(PointsInCircum(1.,3)[:3], transform_to_nD)).float()
data_val = torch.from_numpy(PointsInCircumNDim(PointsInCircum(1.,200), transform_to_nD)).float()

print('PointsInCircum(1.,3)', PointsInCircum(1.,2))

print('len(PointsInCircum(1.,3))', len(PointsInCircum(1.,2)))

#print('data_tr.shape', data_tr.shape)

print('data_tr', data_tr)

loader_tr = get_loader(data_tr)
loader_val = get_loader(data_val)
##################################################################################################################
model = AE(dim, 6, 2, 2, Sin()).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
##################################################################################################################
'''model_conv = ConvAE(dim, no_filters=5, no_layers=2, 
                    kernel_size=3, latent_dim=2, activation=Sin()).to(device)
pytorch_total_params = sum(p.numel() for p in model_conv.parameters() if p.requires_grad)
print(pytorch_total_params)'''
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
latent_dim = 2

model = AE(dim, hidden_size, 2, no_layers, Sin()).to(device)

model_reg_tr = AE(dim, hidden_size, 2, no_layers, Sin()).to(device)
model_reg_ran = AE(dim, hidden_size, 2, no_layers, Sin()).to(device)
model_reg_cheb = AE(dim, hidden_size, 2, no_layers, Sin()).to(device)
model_reg_leg = AE(dim, hidden_size, 2, no_layers, Sin()).to(device)

#model_conv = ConvAE(dim, no_filters, no_layers_conv, kernel_size, latent_dim=2, activation=Sin()).to(device)
#
model_conv = ConvoAE(latent_dim).to(device)

model_contra = Autoencoder_linear(latent_dim).to(device)

model_vae = BetaVAE(dim, no_filters, no_layers_conv,
                  kernel_size, latent_dim=2, activation = Sin(), beta=0.01, use_mu=1.).to(device)
#model_mlp_vae = MLPVAE(dim, hidden_size, 2, no_layers, beta=0.01).to(device)
#
model_mlp_vae = VAE_mlp_circle_new(image_size=15, h_dim=6, z_dim=latent_dim).to(device)
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

#optimizer_mlp_vae = torch.optim.Adam(model_mlp_vae.parameters(), lr=1e-3)
#
optimizer_mlp_vae = torch.optim.Adam(model_mlp_vae.parameters(), lr=1e-3) 

#optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=5e-3)
#
optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr =0.002, weight_decay = 1e-5)

optimizer_contra = torch.optim.Adam(model_contra.parameters(), lr =0.002, weight_decay = 1e-5)

optimizer_vae = torch.optim.Adam(model_vae.parameters(), lr=5e-3)

mod_loss = []
mod_loss_tr = []
mod_loss_ran = []
mod_loss_cheb = []
mod_loss_leg = []
mod_loss_conv = []
mod_loss_vae = []
mod_loss_mlp_vae = []

'''for epoch in range(no_epochs):
    optimizer.zero_grad()
    model_output = model(data_tr.to(device))
    loss = torch.nn.MSELoss()(model_output, data_tr.to(device))
    mod_loss.append(float(loss.item()))
    loss.backward()
    optimizer.step()'''
    
'''plt.plot(list(range(0,no_epochs)), mod_loss, label='baseline, '+str(mod_loss[-1]))
plt.xlabel("$epoch$")
plt.ylabel("$loss$")
plt.title("Baseline model")
plt.grid(True)
plt.show()
plt.savefig('/home/ramana44/autoencoder-regularisation-/circle_experiments_previous/imagesaves/baselineLoss.png')
plt.close()'''
from regularisers_without_vegas import computeC1Loss, sampleChebyshevNodes, sampleLegendreNodes

'''regNodesSamplings = (["trainingData", "random", "chebyshev", "legendre",
                    "conv", "vae", "mlp_vae"])'''
                    
regNodesSamplings = (["mlp_ae", "trainingData", "random", "chebyshev", "legendre",
                    "conv", "contra", "mlp_vae"])

'''models = ([model_reg_tr, model_reg_ran, model_reg_cheb, model_reg_leg,
        model_conv, model_vae, model_mlp_vae])'''

models = ([model, model_reg_tr, model_reg_ran, model_reg_cheb, model_reg_leg,
        model_conv, model_contra, model_mlp_vae])

'''optimizers = ([optimizer_tr, optimizer_ran, optimizer_cheb, optimizer_leg, 
            optimizer_conv, optimizer_contra,  optimizer_vae, optimizer_mlp_vae])'''

optimizers = ([optimizer, optimizer_tr, optimizer_ran, optimizer_cheb, optimizer_leg, 
            optimizer_conv, optimizer_contra, optimizer_mlp_vae])

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

        if (regNodesSampling != "conv") and (regNodesSampling != "vae") and (regNodesSampling != "mlp_vae") and (regNodesSampling != "contra") and (regNodesSampling != "mlp_ae") and (regNodesSampling != "mlp_vae"):
                        
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
        
        if regNodesSampling == "mlp_ae":
            model_output = model_reg(data_tr.to(device))
            loss = torch.nn.MSELoss()(model_output, data_tr.to(device))
            mod_loss_reg.append(float(loss.item()))

        if regNodesSampling == "conv":
            model_output = model_reg(data_tr.unsqueeze(1).to(device))
            loss = torch.nn.MSELoss()(model_output.squeeze(1), data_tr.to(device))
            mod_loss_reg.append(float(loss.item()))

        if regNodesSampling == "contra":
            lam = 1e-2
            img = data_tr.unsqueeze(1).to(device)
            img = data_tr.to(device)
            img = Variable(img)
            recon = model_reg(img)
            W = list(model_reg.parameters())[6]
            hidden_representation = model_reg.encoder(img)
            loss, testcontraLoss = contractive_loss_function(W, img, recon,
                                hidden_representation, lam)
            mod_loss_reg.append(float(loss.item()))
        
        if regNodesSampling == "mlp_vae":
            #print('data_tr.shape', data_tr.shape)
            #images = data_tr.reshape(-1, 15)
            images = data_tr
            recon_images, mu, logvar = model_reg(images.float().to(device))
            #print('recon_images.shape', recon_images.shape)
            loss, bce, kld = loss_fn_mlp_vae(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))
            
            print("mlp vae loss", loss)
            mod_loss_reg.append(float(loss.item()))


        if regNodesSampling == "vae":
            x_reco, mu, logvar = model_reg(data_tr.to(device))
            recon_loss = torch.nn.MSELoss()(data_tr.to(device), x_reco.squeeze(1))
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss = recon_loss + model_vae.beta*total_kld
            mod_loss_reg.append(float(recon_loss.item()))
        if regNodesSampling == "mlp_vae_":
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

labels = ["mlp_ae", "Reg on training data", "Reg on random points", "Reg on chebyshev nodes", "Reg on legendre nodes","conv", "contra", "mlp_vae"]

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

    elif (labels[ind] == "conv" or labels[ind] == "contra" ):
        print('ind', ind)
        points_tr = (model_reg.encoder(data_tr.unsqueeze(1).to(device))).detach().cpu().numpy()
        points_val = (model_reg.encoder(data_val.unsqueeze(1).to(device))).detach().cpu().numpy()
        points_tr = points_tr.reshape(-1,2)
        points_val = points_val.reshape(-1,2)

        plt.scatter(points_val[:,0], points_val[:,1], label='validation samples', color="orange")
        plt.scatter(points_tr[:,0], points_tr[:,1], label='training samples', color="blue")
        plt.scatter(arr_points[:,0], arr_points[:,1], color='grey', alpha=0.05)
        plt.grid(True)
        plt.title(labels[ind])
        plt.show()
        plt.savefig('/home/ramana44/autoencoder-regularisation-/circle_experiments_previous/imagesaves/s'+str(ind)+'.png')
        plt.close()
    elif labels[ind] == "mlp_vae":
        #points_tr, _, _ = (model_reg.encode(data_tr.to(device), False))
        points_tr = model_reg.fc1(model_reg.encoder(data_tr.float().to(device)))
        points_tr = points_tr.detach().cpu().numpy()

        #points_val,_ ,_ = (model_reg.encode(data_val.to(device), False))
        points_val = model_reg.fc1(model_reg.encoder(data_val.float().to(device)))
        points_val = points_val.detach().cpu().numpy()

        points_tr = points_tr.reshape(-1,2)
        points_val = points_val.reshape(-1,2)

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
