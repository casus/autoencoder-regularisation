import sys
sys.path.append('./')

import torch
import torch.nn.functional as F
import os
from models import MLP_VAE_FMNIST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 10
z_dim = latent_dim
TDA = 0.4
h_dim=100
image_size=32*32
lr = 0.0001
no_layers = 3
no_epochs= 100
hidden_size = 100




model = MLP_VAE_FMNIST(image_size, h_dim, latent_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

batch_size_cfs = 200
image_batches_trn = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/trainImages.pt').to(device)
image_batches_trn = image_batches_trn.reshape(int(image_batches_trn.shape[0]/batch_size_cfs), batch_size_cfs, 1, 32,32)
image_batches_test = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/testImages.pt').to(device)
image_batches_test = image_batches_test.reshape(int(image_batches_test.shape[0]/batch_size_cfs), batch_size_cfs, 1, 32,32)
image_batches_trn = image_batches_trn[:int(image_batches_trn.shape[0]*TDA)]
image_batches_test = image_batches_test[:int(image_batches_test.shape[0]*TDA)]

epochs = 50

for epoch in range(epochs):
    inum = 0
    for images in image_batches_trn:    
        inum = inum+1
        images = images.reshape(-1, 32*32)

        recon_images, mu, logvar = model(images.to(device))

        loss, bce, kld = loss_fn(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch : ", epoch)


path = './models_saved/'
os.makedirs(path, exist_ok=True)


name = '_'+reg_nodes_sampling+'_'+'_'+str(TDA)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)
torch.save(model.state_dict(), path+'/model_base_mlp_vae_TDA'+name)

