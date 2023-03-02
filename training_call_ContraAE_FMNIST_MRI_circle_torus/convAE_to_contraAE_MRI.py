import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib.pyplot import plot
import os
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()

#FashionMNISt_data = datasets.FashionMNIST(root= './data', train=True, download=True, transform=transform.Resize(96) )

#data_loader = torch.utils.data.DataLoader(dataset = FashionMNISt_data, batch_size = 200, shuffle = True)

no_layers = 5
latent_dim = 40
batch_size = 200
TDA = 0.005
lr =1e-3
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

class Autoencoder_linear(nn.Module):
    def __init__(self,latent_dim):

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(96*96, 1000),  #input layer
            nn.ReLU(),
            nn.Linear(1000, 1000),   #h1
            nn.ReLU(),
            nn.Linear(1000, 1000),    #h1
            nn.ReLU(),
            nn.Linear(1000, 1000),    #h1
            nn.ReLU(),
            nn.Linear(1000, 1000),    #h1
            nn.ReLU(),
            nn.Linear(1000, 1000),    #h1
            nn.ReLU(),
            nn.Linear(1000,latent_dim)  # latent layer
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1000),  #input layer
            nn.ReLU(),
            nn.Linear(1000, 1000),   #h1
            nn.ReLU(),
            nn.Linear(1000, 1000),   #h1
            nn.ReLU(),
            nn.Linear(1000, 1000),   #h1
            nn.ReLU(),
            nn.Linear(1000, 1000),    #h1
            nn.ReLU(),
            nn.Linear(1000, 1000),    #h1
            nn.ReLU(),
            nn.Linear(1000, 96*96),  # latent layer
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(100, 80, bias=False)
        init.normal(self.fc1.weight, mean=0, std=1)
        self.fc2 = nn.Linear(80, 87)
        self.fc3 = nn.Linear(87, 94)
        self.fc4 = nn.Linear(94, 100)

    def forward(self, x):
         x = self.fc1(x)
         x = F.relu(self.fc2(x))
         x = F.relu(self.fc3(x))
         x = F.relu(self.fc4(x))
         return x

class ConvAE(nn.Module):
    def __init__(self):
        # 1 as input in Conv2d indicates the number of channels
        super().__init__()
        #N, 1, 96, 96
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  #N, 16, 16, 16
            nn.ReLU(),
            nn.Conv2d(16, 96, 3, stride=2, padding=1),   #N, 96, 8, 8
            nn.ReLU(),
            nn.Conv2d(96, 64, 8)    #N, 64, 1, 1

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 96, 8),  #N, 96, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(96, 16, 3, stride=2, padding=1, output_padding=1),   #N, 16, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   #N, 1, 96, 96
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder_linear(latent_dim).to(device)
mseLoss_nn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =lr, weight_decay = weight_decay)

mse_loss = nn.BCELoss(size_average = False)

def loss_function(W, x, recons_x, h, lam):
    """Compute the Contractive AutoEncoder Loss
    Evalutes the CAE loss, which is composed as the summation of a Mean
    Squared Error and the weighted l2-norm of the Jacobian of the hidden
    units with respect to the inputs.
    See reference below for an in-depth discussion:
      #1: http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder
    Args:
        `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
          dimensions of the hidden units and input respectively.
        `x` (Variable): the input to the network, with dims (N_batch x N)
        recons_x (Variable): the reconstruction of the input, with dims
          N_batch x N.
        `h` (Variable): the hidden units of the network, with dims
          batch_size x N_hidden
        `lam` (float): the weight given to the jacobian regulariser term
    Returns:
        Variable: the (scalar) CAE loss
    """
    mse = mse_loss(recons_x, x)
    #mse = mseLoss_nn(recons_x, x)

    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W)**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)

    #print('contractive_loss.mul_(lam)', contractive_loss.mul_(lam))
    #print('mse', mse)
    return mse + contractive_loss.mul_(lam), contractive_loss.mul_(lam)
num_epochs = 100 
outputs = []


#test_model = nn.Conv2d(1, 16, 3, stride=2, padding=1),  #N, 16, 14, 14
#test_model = nn.Conv2d(3, 6, 5),  #N, 16, 14, 14

lam = 1e-3

for epoch in range(num_epochs):
    #print('epochs', epochs)
    for img in image_batches_trn:
        #print('img.shape', img.shape)
        img = img.reshape(-1, 96*96)
        #print('img.shape', img.shape)

        #test_it = test_model(img)

        #print('test_it.shape', test_it.shape)
        img = Variable(img)


        recon = model(img)
        #print('recon.shape', recon.shape)
        #print('list(model.parameters())[0].shape', list(model.parameters())[8].shape)   

        #print('list(model.parameters()).shape', len(list(model.encoder.parameters())))   

        W = list(model.parameters())[12]
        #print('W.shape', W.shape)
        hidden_representation = model.encoder(img)
        #print('hidden_representation', hidden_representation.shape)

        loss, testcontraLoss = loss_function(W, img, recon,
                             hidden_representation, lam)

        #break     
        #loss = criterion(recon, img)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #break

    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, ContraLoss: {testcontraLoss.item():.4f}')
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
torch.save(model.state_dict(), path+'/model_base_contraAE_MRI'+name)