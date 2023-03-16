import sys
sys.path.append('./')

import torch
import torch.nn as nn
import os
from models import ConvoAE_mri


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

no_layers = 5
latent_dim = 40
batch_size = 200
TDA = 0.05
lr =1e-4
weight_decay = 1e-5

image_batches_trn = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/trainDataSet.pt',map_location=torch.device('cuda'))
image_batches_trn = image_batches_trn.reshape(int(image_batches_trn.shape[0]/batch_size), batch_size, 1, 96,96)
image_batches_trn = image_batches_trn[:int(image_batches_trn.shape[0]*TDA)]

model = ConvoAE_mri(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =lr, weight_decay = weight_decay)


num_epochs = 100 
outputs = []


for epoch in range(num_epochs):
    for img in image_batches_trn:

        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')
    outputs.append((epoch, img, recon))

path = './models_saved/'
os.makedirs(path, exist_ok=True)
name = '_'+str(TDA)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)
torch.save(model.state_dict(), path+'/model_base_cae_MRI'+name)


