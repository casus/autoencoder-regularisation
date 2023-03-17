import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary
import os
from pushover import notify
from utils import makegif
from random import randint

from IPython.display import Image
from IPython.core.display import Image, display

from vae import VAE_mlp_circle_new

#%load_ext autoreload
#%autoreload 2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bs = 32

train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.Resize(32),transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.Resize(32),transforms.ToTensor()])) 

train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=200)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=200)

fixed_x, _ = next(iter(train_loader))
image_channels = fixed_x.size(1)


latent_dim = 2
z_dim = latent_dim
TDA = 1.0
h_dim=6
image_size=15

model = VAE_mlp_circle_new(image_size, h_dim, latent_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.float(), x.float(), size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

batch_size_cfs = 1

image_batches_trn = torch.load('/home/ramana44/autoencoder-regularisation-/circle_dataset_no_normalization/circleThreeTrainingPointsIn15D.pt').to(device)

image_batches_trn = image_batches_trn.reshape(int(image_batches_trn.shape[0]/batch_size_cfs), batch_size_cfs, 1, 15)


epochs = 70

for epoch in range(epochs):
    #for idx, (images, _) in enumerate(train_loader):
    inum = 0
    for images in image_batches_trn:    
        inum = inum+1
        images = images.reshape(1, 15)

        #images = torch.cuda.FloatTensor(images)
        #print()
        #print('model.encoder(images.to(device)).shape',model.encoder(images.to(device)).shape)
        #print('images.shape', images.shape)
        #images = images.reshape(200, 1, 1, 32, 32)
        #this = model.encoder(images.to(device))
        #print('this.shape', this.shape)
        #print('model.fc1(this).shape',model.fc1(this).shape )
        recon_images, mu, logvar = model(images.float().to(device))

        #print('recon_images.shape', recon_images.shape)



        loss, bce, kld = loss_fn(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                #epochs, loss.data[0]/bs, bce.data[0]/bs, kld.data[0]/bs)
        #print(to_print)

    print("Epoch : ", epoch)

# notify to android when finished training
#notify(to_print, priority=1)

#torch.save(model.state_dict(), 'vae.torch')


reg_nodes_sampling = 'legendre'
deg_poly = 21
lr = 0.0001
no_layers = 2
no_epochs= 100
alpha = 0.5
hidden_size = 6
#latent_dim = 
path = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'
#path = './output/MRT_full/test_run_saving/'
os.makedirs(path, exist_ok=True)


name = '_'+reg_nodes_sampling+'_'+'_'+str(TDA)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)
#torch.save(loss_arr_reg, path+'/loss_arr_reg_cae_TDA'+name)
#torch.save(loss_arr_reco, path+'/loss_arr_reco_cae_TDA'+name)
#torch.save(loss_arr_base, path+'/loss_arr_base_bvae_TDA'+name)
#torch.save(loss_arr_val_reco, path+'/loss_arr_val_reco_cae_TDA'+name)
#torch.save(loss_arr_val_base, path+'/loss_arr_val_base_bvae_TDA'+name)
torch.save(model.state_dict(), path+'/model_base_mlp_vae_circle'+name)
#torch.save(model_reg.state_dict(), path+'/model_reg_cae_TDA'+name)
print("is it saved")
