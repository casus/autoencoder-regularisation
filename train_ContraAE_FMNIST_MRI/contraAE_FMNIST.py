import sys
sys.path.append('./')

import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from models import Autoencoder_linear_contra_fmnist
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


no_layers = 3
latent_dim = 10
batch_size = 200
TDA = 0.1
lr =1e-3
weight_decay = 1e-5
num_epochs = 100


image_batches_trn = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/trainImages.pt').to(device)
image_batches_trn = image_batches_trn.reshape(int(image_batches_trn.shape[0]/batch_size), batch_size, 1, 32,32)
image_batches_test = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/testImages.pt').to(device)
image_batches_test = image_batches_test.reshape(int(image_batches_test.shape[0]/batch_size), batch_size, 1, 32,32)
image_batches_trn = image_batches_trn[:int(image_batches_trn.shape[0]*TDA)]
image_batches_test = image_batches_test[:int(image_batches_test.shape[0]*TDA)]



model = Autoencoder_linear_contra_fmnist(latent_dim).to(device)
mseLoss_nn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =lr, weight_decay = weight_decay)

mse_loss = nn.BCELoss(size_average = False)

def loss_function(W, x, recons_x, h, lam):

    mse = mse_loss(recons_x, x)
    dh = h * (1 - h) 
    w_sum = torch.sum(Variable(W)**2, dim=1)
    w_sum = w_sum.unsqueeze(1) 
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam), contractive_loss.mul_(lam)
outputs = []


lam = 1e-2

loss_track = []
for epoch in range(num_epochs):
    for img in image_batches_trn:
        img = img.reshape(-1, 32*32)
        img = Variable(img)
        recon = model(img)
        W = list(model.parameters())[8]
        hidden_representation = model.encoder(img)
        loss, testcontraLoss = loss_function(W, img, recon, hidden_representation, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, ContraLoss: {testcontraLoss.item():.4f}')
    outputs.append((epoch, img, recon))
    loss_track.append(loss)

path = './models_saved/'
os.makedirs(path, exist_ok=True)
name = '_'+str(TDA)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)
torch.save(model.state_dict(), path+'/model_base_contraAE_TDA'+name)
