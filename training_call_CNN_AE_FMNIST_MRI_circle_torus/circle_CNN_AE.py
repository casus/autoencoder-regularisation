import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib.pyplot import plot
import os
from activations import Sin


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()

#FashionMNISt_data = datasets.FashionMNIST(root= './data', train=True, download=True, transform=transform.Resize(32) )

#data_loader = torch.utils.data.DataLoader(dataset = FashionMNISt_data, batch_size = 200, shuffle = True)

no_layers = 2
latent_dim = 2
batch_size = 200
TDA = 1.0
lr =0.002
weight_decay = 1e-5


batch_size_cfs = 1
#coeffs_saved_trn = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/AllFmnistTrainRKCoeffsDeg25.pt').to(device)
image_batches_trn = torch.load('/home/ramana44/autoencoder-regularisation-/circle_dataset_no_normalization/circleThreeTrainingPointsIn15D.pt').to(device)


image_batches_trn = image_batches_trn.reshape(int(image_batches_trn.shape[0]/batch_size_cfs), batch_size_cfs, 1, 15)
#coeffs_saved_trn = coeffs_saved_trn.reshape(int(coeffs_saved_trn.shape[0]/batch_size_cfs), batch_size_cfs, coeffs_saved_trn.shape[1]).unsqueeze(2) 


#coeffs_saved_test = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/AllFmnistTestRKCoeffsDeg25.pt').to(device)
image_batches_test = torch.load('/home/ramana44/autoencoder-regularisation-/circle_dataset_no_normalization/circleThreeTrainingPointsIn15D.pt').to(device)
#image_batches_test = image_batches_test[:11200]
image_batches_test = image_batches_test.reshape(int(image_batches_test.shape[0]/batch_size_cfs), batch_size_cfs, 1, 15)
#coeffs_saved_test = coeffs_saved_test[:11200]
#coeffs_saved_test = coeffs_saved_test.reshape(int(coeffs_saved_test.shape[0]/batch_size_cfs), batch_size_cfs, coeffs_saved_test.shape[1]).unsqueeze(2) 


#print('coeffs_saved_trn.shape',coeffs_saved_trn.shape)



#print('coeffs_saved_test.shape',coeffs_saved_test.shape)
image_batches_trn = image_batches_trn[:int(image_batches_trn.shape[0]*TDA)]
image_batches_test = image_batches_test[:int(image_batches_test.shape[0]*TDA)]

print('image_batches_trn.shape',image_batches_trn.shape)
print('image_batches_test.shape',image_batches_test.shape)

class Autoencoder_linear(nn.Module):
    def __init__(self):

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32*32, 100),  #input layer
            nn.ReLU(),
            nn.Linear(100, 100),   #h1
            nn.ReLU(),
            nn.Linear(100, 100),    #h1
            nn.ReLU(),
            nn.Linear(100, 100),    #h1
            nn.ReLU(),
            nn.Linear(100,latent_dim),  # latent layer
            
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 100),  #input layer
            nn.ReLU(),
            nn.Linear(100, 100),   #h1
            nn.ReLU(),
            nn.Linear(100, 100),    #h1
            nn.ReLU(),
            nn.Linear(100, 100),    #h1
            nn.ReLU(),
            nn.Linear(100, 32*32),  # latent layer
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

'''class ConvAE(nn.Module):
    def __init__(self):
        # 1 as input in Conv2d indicates the number of channels
        super().__init__()
        #N, 1, 32, 32
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  #N, 16, 16, 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),   #N, 32, 8, 8
            nn.ReLU(),
            nn.Conv2d(32, 64, 8)    #N, 64, 1, 1

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 8),  #N, 32, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   #N, 16, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   #N, 1, 32, 32
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded'''

class ConvoAE(nn.Module):
    def __init__(self, latent_dim):
        # 1 as input in Conv2d indicates the number of channels
        super().__init__()
        #N, 1, 32, 32
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 5, 3, stride=2, padding=1),  #N, 16, 16, 16
            Sin(),
            nn.Conv1d(5, 5, 3, stride=2, padding=1),   #N, 32, 8, 8
            Sin(),
            nn.Conv1d(5, 5, 3, stride=1, padding=1),   #N, 32, 8, 8
            Sin(),            
            nn.Flatten(1,-1),
            nn.Linear(5*4, latent_dim),
            Sin()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 5*4),
            nn.Unflatten(1, (5, 4)),
            nn.ConvTranspose1d(5, 5, 3, stride=2, padding=1, output_padding=1),  #N, 32, 8, 8
            Sin(),
            nn.ConvTranspose1d(5, 5, 3, stride=1, padding=1, output_padding=0),  #N, 32, 8, 8
            Sin(),
            nn.ConvTranspose1d(5, 1, 3, stride=2, padding=1, output_padding=0),  #N, 32, 8, 8
            Sin()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = ConvoAE(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =lr, weight_decay = weight_decay)


num_epochs = 300
outputs = []


#test_model = nn.Conv2d(1, 16, 3, stride=2, padding=1),  #N, 16, 14, 14
#test_model = nn.Conv2d(3, 6, 5),  #N, 16, 14, 14



for epoch in range(num_epochs):
    #print('epochs', epochs)
    for img in image_batches_trn:
        #print('img.shape', img.shape)
        #img = img.reshape(-1, 32*32)
        #print('img.shape', img.shape)

        #test_it = test_model(img)

        #print('test_it.shape', test_it.shape)
        #print('list(model.encoder.parameters())[8]',list(model.encoder.parameters())[1].shape)
        #break
        img = img.float()
        
        #if(epoch==99):
            #print('model.encoder(img).max', model.encoder(img).max())
            #print('model.encoder(img).min', model.encoder(img).min())
            #break

        recon = model(img)
        #print('recon.shape', recon.shape) 
        loss = criterion(recon, img)
       

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #break

    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')
    outputs.append((epoch, img, recon))

path = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'
#path = './output/MRT_full/test_run_saving/'
os.makedirs(path, exist_ok=True)
name = '_'+str(TDA)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)
#torch.save(loss_arr_reg, path+'/loss_arr_reg_cae_TDA'+name)
#torch.save(loss_arr_reco, path+'/loss_arr_reco_cae_TDA'+name)
#torch.save(loss_arr_base, path+'/loss_arr_base_cae_TDA'+name)
#torch.save(loss_arr_val_reco, path+'/loss_arr_val_reco_cae_TDA'+name)
#torch.save(loss_arr_val_base, path+'/loss_arr_val_base_cae_TDA'+name)
torch.save(model.state_dict(), path+'/model_base_cae_circ'+name)