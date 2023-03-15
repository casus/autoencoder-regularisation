import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib.pyplot import plot
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()

#FashionMNISt_data = datasets.FashionMNIST(root= './data', train=True, download=True, transform=transform.Resize(32) )

#data_loader = torch.utils.data.DataLoader(dataset = FashionMNISt_data, batch_size = 200, shuffle = True)

no_layers = 5
latent_dim = 10
batch_size = 200
TDA = 0.05
lr =1e-4
weight_decay = 1e-5

image_batches_trn = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/trainDataSet.pt',map_location=torch.device('cuda'))

#image_batches_trn = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/trainImages.pt').to(device)


image_batches_trn = image_batches_trn.reshape(int(image_batches_trn.shape[0]/batch_size), batch_size, 1, 96,96)

#image_batches_test = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/testDataSet.pt',map_location=torch.device('cuda'))
#image_batches_test = image_batches_test.reshape(int(image_batches_test.shape[0]/batch_size), batch_size, 1, 96,96)

image_batches_trn = image_batches_trn[:int(image_batches_trn.shape[0]*TDA)]
#image_batches_test = image_batches_test[:int(image_batches_test.shape[0]*TDA)]

print('image_batches_trn.shape',image_batches_trn.shape)
#print('image_batches_test.shape',image_batches_test.shape)



print(torch.min(image_batches_trn), torch.max(image_batches_trn))

print(image_batches_trn.shape)



class ConvoAE(nn.Module):
    def __init__(self, latent_dim):
        # 1 as input in Conv2d indicates the number of channels
        super().__init__()
        #N, 1, 32, 32
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  #N, 1, 32, 32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  #N, 1, 32, 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  #N, 1, 32, 32
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, stride=1, padding=1),  #N, 1, 32, 32
            nn.ReLU(),
            nn.Conv2d(32, 16, 4, stride=1, padding=1),  #N, 1, 32, 32
            nn.ReLU(),
            nn.Flatten(1,-1),
            nn.Linear(16*9*9, latent_dim)

        )

        self.decoder = nn.Sequential(
            
            nn.Linear(latent_dim, 16*9*9),
            nn.Unflatten(1, (16, 9, 9)),
            nn.ConvTranspose2d(16, 32, 4, stride=1, padding=1),   #N, 1, 32, 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, 5, stride=1, padding=1),   #N, 1, 32, 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   #N, 1, 32, 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   #N, 1, 32, 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   #N, 1, 32, 32
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

'''class ConvoAE(nn.Module):
    def __init__(self):
        # 1 as input in Conv2d indicates the number of channels
        super().__init__()
        #N, 1, 32, 32
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  #N, 16, 16, 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),   #N, 32, 8, 8
            nn.ReLU(),
            nn.Conv2d(32, 1, 8),    #N, 64, 2, 2
            #nn.Flatten()
        )

        self.decoder = nn.Sequential(
            #nn.Unflatten(1,(64,2,2)),
            nn.ConvTranspose2d(1, 32, 8),  #N, 32, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   #N, 16, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   #N, 1, 32, 32
            nn.Sigmoid()
        )



'''



model = ConvoAE(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =lr, weight_decay = weight_decay)


num_epochs = 100 
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

        #print('list(model.encoder.parameters())[8]',list(model.encoder.parameters())[2].shape)

        #print('img.shape', img.shape)
        #print('model.encoder(img)', model.encoder(img).shape)
        recon = model(img)
        #print('recon.shape', recon.shape)
        #print('img.shape', img.shape)

        #print('model.encoder(img)', model.encoder(img).shape)

        #break

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
torch.save(model.state_dict(), path+'/model_base_cae_MRI'+name)


