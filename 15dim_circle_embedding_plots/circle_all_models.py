import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib.pyplot import plot
import os
from activations import Sin

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()



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
            nn.Linear(100,latent_dim)  # latent layer
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