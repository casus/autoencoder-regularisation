import sys
sys.path.append('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders')


from get_data import get_data, get_data_train, get_data_val
import torch
import os
import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline

from datasets import InMemDataLoader
import torch.nn.functional as F
import torch
import nibabel as nib     # Read / write access to some common neuroimaging file formats
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
from scipy import interpolate
import ot

import jmp_solver1.surrogates
import matplotlib
matplotlib.rcdefaults() 



deg_quad = 16
u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
x = np.linspace(-1,1,32)
X_p = u_ob.data_axes([x,x]).T

def get_all_thetas(listedImage):
    get = np.linalg.lstsq(np.array(X_p), listedImage.reshape(32*32), rcond='warn')
    act_theta = torch.tensor(get[0])
    return act_theta





# load trained rAE and bAE

#from models import AE_un
from models import AE
from activations import Sin

path_hyb = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_space/output/MRT_full/test_run_saving/'
path_unhyb = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'

#specify hyperparameters
reg_nodes_sampling = 'legendre'
alpha = 0.5
frac = 0.4
hidden_size = 100
deg_poly = 21
deg_poly_forRK = 21
latent_dim = 2
lr = 0.0001
no_layers = 3
no_epochs= 100
name_hyb = '_'+reg_nodes_sampling+'_'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly_forRK)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)
name_unhyb = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

#no_channels, dx, dy = (train_loader_alz.dataset.__getitem__(1).shape)
#inp_dim = [no_channels, dx-21, dy-21]
inp_dim_hyb = (deg_quad+1)*(deg_quad+1)

inp_dim_unhyb = [1,32,32]

RK_model_reg = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
RK_model_base = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

model_reg = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
model_base = AE(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

#model_reg.load_state_dict(torch.load(path+'model_reg'+name, map_location=torch.device('cpu'))["model"])
#model_base.load_state_dict(torch.load(path+'model_reg'+name, map_location=torch.device('cpu'))["model"])

RK_model_reg.load_state_dict(torch.load(path_hyb+'model_regLSTQS'+str(deg_quad)+''+name_hyb, map_location=torch.device('cpu')))
RK_model_base.load_state_dict(torch.load(path_hyb+'model_baseLSTQS'+str(deg_quad)+''+name_hyb, map_location=torch.device('cpu')))

model_reg.load_state_dict(torch.load(path_unhyb+'model_reg_TDA'+name_unhyb, map_location=torch.device('cpu')))
model_base.load_state_dict(torch.load(path_unhyb+'model_base_TDA'+name_unhyb, map_location=torch.device('cpu')))
#model_reg.eval()
#model_base.eval()

print("Anything here to see")


#loading convolutional autoencoder
from convAE import ConvoAE
no_layers_cae = 3
latent_dim_cae = latent_dim
lr_cae =1e-3
name_unhyb_cae = '_'+str(frac)+'_'+str(latent_dim_cae)+'_'+str(lr_cae)+'_'+str(no_layers_cae)
model_convAE = ConvoAE(latent_dim_cae).to(device)
model_convAE.load_state_dict(torch.load(path_unhyb+'model_base_cae_TDA'+name_unhyb_cae, map_location=torch.device(device)), strict=False)


#rec = model_convAE(trainImages).view(trainImages.shape).detach().numpy() 

#from vae import BetaVAE
from activations import Sin
activation = Sin()
from vae_models_for_fmnist import VAE_try, VAE_mlp, Autoencoder_linear
#model_mlpVAE = MLPVAE(1*32*32, hidden_size, latent_dim, 
                #    no_layers, activation).to(device) # regularised autoencoder

model_mlpVAE_ = VAE_mlp(32*32, hidden_size, latent_dim).to(device)

#model_betaVAE = BetaVAE([32, 32], 1, no_filters=4, no_layers=3,
                #kernel_size=3, latent_dim=10, activation = Sin()).to(device) # regularised autoencoder

model_cnnVAE_ = VAE_try(image_channels=1, h_dim=8*2*2, z_dim=latent_dim).to(device)

#model_betaVAE = BetaVAE(batch_size = 1, img_depth = 1, net_depth = no_layers, z_dim = latent_dim, img_dim = 32).to(device)
model_mlpVAE_.load_state_dict(torch.load(path_unhyb+'model_base_mlp_vae_TDA'+name_unhyb, map_location=torch.device(device)), strict=False)
model_cnnVAE_.load_state_dict(torch.load(path_unhyb+'model_base_cnn_vae_TDA'+name_unhyb, map_location=torch.device(device)), strict=False)

def model_mlpVAE(input):
    #print('model_betaVAE(input).shape', model_betaVAE(input).shape)
    input = input.reshape(-1, 32*32)
    recon= model_mlpVAE_.fc1(model_mlpVAE_.encoder(input))
    return recon

def model_cnnVAE(input):
    #print('model_betaVAE(input).shape', model_betaVAE(input).shape)
    recon= model_cnnVAE_.fc1(model_cnnVAE_.encoder(input))
    return recon

'''def model_contra(input):
    input = input.reshape(-1, 32*32)
    recon= model_cnnVAE_.encoder(input)
    return recon'''

#rec = model_convAE(torch.from_numpy(trainImages).reshape(1,1,32,32).to(device))

#loading contractive autoencoder
no_layers_contraae = 3
latent_dim_contraae = latent_dim
lr_contraae =1e-3
name_unhyb_contraae = '_'+str(frac)+'_'+str(latent_dim_contraae)+'_'+str(lr_contraae)+'_'+str(no_layers_contraae)
model_contra_ = Autoencoder_linear(latent_dim).to(device)
model_contra_.load_state_dict(torch.load(path_unhyb+'model_base_contraAE_TDA'+name_unhyb_contraae, map_location=torch.device(device)), strict=False)

def model_contra(input):
    input = input.reshape(-1, 32*32)
    recon= model_contra_.encoder(input)
    return recon

print()
print("anything ? all ok")

Analys_size = 1000
colors = ["red","green","blue","yellow","cyan","black","orange","magenta","brown","gray"]
FMNISTlabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for i in range(len(colors)):
    testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label'+str(i)+'.pt',map_location=torch.device('cpu'))
    testImages = testImages[:Analys_size]
    plt.imshow(testImages[0][0])
    plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNIST2d_latent_space/classes_egs/eg'+str(i)+'.png')   
plt.close()

for i in range(len(colors)):
    testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label'+str(i)+'.pt',map_location=torch.device('cpu'))
    testImages = testImages[:Analys_size]
    unhyb_rec_rAE_test = model_reg.encoder(testImages).detach().numpy()
    unhyb_rec_rAE_test = torch.tensor(unhyb_rec_rAE_test, requires_grad=False)
    #unhyb_rec_rAE_test = -1 + ( ( ( unhyb_rec_rAE_test - unhyb_rec_rAE_test.min() ) * 2 ) / ( unhyb_rec_rAE_test.max() - unhyb_rec_rAE_test.min() ) )
    #print('unhyb_rec_rAE_test.max()', unhyb_rec_rAE_test.max())
    #print('unhyb_rec_rAE_test.min()', unhyb_rec_rAE_test.min())
    plt.scatter(unhyb_rec_rAE_test[:,0], unhyb_rec_rAE_test[:,1], color=colors[i])
    plt.xlim(-0.8, 1.0)
    plt.legend(FMNISTlabels)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNIST2d_latent_space/plots/AE_REG.png')
plt.close()

unhyb_rec_bAE_test = model_base.encoder(testImages).detach().numpy() 
unhyb_rec_bAE_test = torch.tensor(unhyb_rec_bAE_test, requires_grad=False)

for i in range(len(colors)):
    testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label'+str(i)+'.pt',map_location=torch.device('cpu'))
    testImages = testImages[:Analys_size]
    unhyb_rec_bAE_test = model_base.encoder(testImages).detach().numpy() 
    unhyb_rec_bAE_test = torch.tensor(unhyb_rec_bAE_test, requires_grad=False)
    #unhyb_rec_bAE_test = -1 + ( ( ( unhyb_rec_bAE_test - unhyb_rec_bAE_test.min() ) * 2 ) / ( unhyb_rec_bAE_test.max() - unhyb_rec_bAE_test.min() ) )

    plt.scatter(unhyb_rec_bAE_test[:,0], unhyb_rec_bAE_test[:,1], color=colors[i], label = str(i))
    plt.xlim(-1.2, 1.4)
    plt.legend(FMNISTlabels)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNIST2d_latent_space/plots/MLP_AE.png')
plt.close()

#convAE_rec_test = model_convAE.encoder(testImages).detach().numpy() 
#convAE_rec_test = torch.tensor(convAE_rec_test, requires_grad=False)

for i in range(len(colors)):
    testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label'+str(i)+'.pt',map_location=torch.device('cpu'))
    testImages = testImages[:Analys_size]
    convAE_rec_test = model_convAE.encoder(testImages).detach().numpy() 
    convAE_rec_test = torch.tensor(convAE_rec_test, requires_grad=False)
    convAE_rec_test = -1 + ( ( ( convAE_rec_test - convAE_rec_test.min() ) * 2 ) / ( convAE_rec_test.max() - convAE_rec_test.min() ) )
    plt.scatter(convAE_rec_test[:,0], convAE_rec_test[:,1], color=colors[i], label = str(i))
    plt.xlim(-1.3, 1.2)
    plt.legend(FMNISTlabels)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNIST2d_latent_space/plots/Conv_AE.png')
plt.close()


#mlpVAE_rec_test = model_mlpVAE(testImages).detach().numpy() 
#mlpVAE_rec_test = torch.tensor(mlpVAE_rec_test, requires_grad=False)

for i in range(len(colors)):
    testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label'+str(i)+'.pt',map_location=torch.device('cpu'))
    testImages = testImages[:Analys_size]
    mlpVAE_rec_test = model_mlpVAE(testImages).detach().numpy() 
    mlpVAE_rec_test = torch.tensor(mlpVAE_rec_test, requires_grad=False)
    mlpVAE_rec_test = -1 + ( ( ( mlpVAE_rec_test - mlpVAE_rec_test.min() ) * 2 ) / ( mlpVAE_rec_test.max() - mlpVAE_rec_test.min() ) )
    plt.scatter(mlpVAE_rec_test[:,0], mlpVAE_rec_test[:,1], color=colors[i], label = str(i))
    plt.xlim(-1.2, 1.4)
    plt.legend(FMNISTlabels)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNIST2d_latent_space/plots/MLP_VAE.png')
plt.close()


#cnnVAE_rec_test = model_cnnVAE(testImages).detach().numpy() 
#cnnVAE_rec_test = torch.tensor(cnnVAE_rec_test, requires_grad=False)

for i in range(len(colors)):
    testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label'+str(i)+'.pt',map_location=torch.device('cpu'))
    testImages = testImages[:Analys_size]
    cnnVAE_rec_test = model_cnnVAE(testImages).detach().numpy() 
    cnnVAE_rec_test = torch.tensor(cnnVAE_rec_test, requires_grad=False)
    cnnVAE_rec_test = -1 + ( ( ( cnnVAE_rec_test - cnnVAE_rec_test.min() ) * 2 ) / ( cnnVAE_rec_test.max() - cnnVAE_rec_test.min() ) )
    plt.scatter(cnnVAE_rec_test[:,0], cnnVAE_rec_test[:,1], color=colors[i], label = str(i))
    plt.xlim(-1.5, 1.2)
    plt.legend(FMNISTlabels)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNIST2d_latent_space/plots/CNN_VAE.png')
plt.close()


contra_rec_test = model_contra(testImages).detach().numpy() 
contra_rec_test = torch.tensor(contra_rec_test, requires_grad=False)

for i in range(len(colors)):
    testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label'+str(i)+'.pt',map_location=torch.device('cpu'))
    testImages = testImages[:Analys_size]
    contra_rec_test = model_contra(testImages).detach().numpy() 
    contra_rec_test = torch.tensor(contra_rec_test, requires_grad=False)
    contra_rec_test = -1 + ( ( ( contra_rec_test - contra_rec_test.min() ) * 2 ) / ( contra_rec_test.max() - contra_rec_test.min() ) )
    plt.scatter(contra_rec_test[:,0], contra_rec_test[:,1], color=colors[i], label = str(i))
    plt.xlim(-1.2, 1.2)
    plt.legend(FMNISTlabels)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNIST2d_latent_space/plots/Contra_AE.png')
plt.close()


'''testCoeffs = torch.tensor([])
for i in range(testImages.shape[0]):
    testCoeffs_cur = get_all_thetas(testImages[i]).unsqueeze(0)
    testCoeffs = torch.cat((testCoeffs, testCoeffs_cur),0)

testCoeffs = testCoeffs[:Analys_size]

rec_rAE_test = RK_model_reg.encoder(testCoeffs.float())
rec_rAE_test = torch.tensor(rec_rAE_test, requires_grad=False)'''

for i in range(len(colors)):
    testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNISTClassifiedTestData/label'+str(i)+'.pt',map_location=torch.device('cpu'))
    testImages = testImages[:Analys_size]

    testCoeffs = torch.tensor([])
    for j in range(testImages.shape[0]):
        testCoeffs_cur = get_all_thetas(testImages[j]).unsqueeze(0)
        testCoeffs = torch.cat((testCoeffs, testCoeffs_cur),0)
    testCoeffs = testCoeffs[:Analys_size]
    rec_rAE_test = RK_model_reg.encoder(testCoeffs.float())
    rec_rAE_test = torch.tensor(rec_rAE_test, requires_grad=False)
    #rec_rAE_test = -1 + ( ( ( rec_rAE_test - rec_rAE_test.min() ) * 2 ) / ( rec_rAE_test.max() - rec_rAE_test.min() ) )
    plt.scatter(rec_rAE_test[:,0], rec_rAE_test[:,1], color=colors[i], label = str(i))
    plt.xlim(-0.5, 1.4)
    plt.legend(FMNISTlabels)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/fashionMNIST2d_latent_space/plots/Hybrid_AE_REG.png')
plt.close()


print()
print('unhyb_rec_rAE_test.shape', unhyb_rec_rAE_test.shape)
print()
print('unhyb_rec_bAE_test.shape', unhyb_rec_bAE_test.shape)
print()
print('convAE_rec_test.shape', convAE_rec_test.shape)
print()
print('mlpVAE_rec_test.shape', mlpVAE_rec_test.shape)
print()
print('cnnVAE_rec_test.shape', cnnVAE_rec_test.shape)
print()
print('contra_rec_test.shape', contra_rec_test.shape)
print()
print('rec_rAE_test.shape', rec_rAE_test.shape)
print()
print("All Ok? ")