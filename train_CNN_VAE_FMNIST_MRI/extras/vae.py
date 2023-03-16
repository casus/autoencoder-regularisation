import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from activations import Sin



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=256):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32*2, 64*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64*2, 128*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128*2, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128*2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128*2, 64*2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64*2, 32*2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32*2, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar

class VAE3232input(nn.Module):
    def __init__(self, image_channels=3, h_dim=128, z_dim=4):
        super(VAE3232input, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=4, stride=2, padding = 1),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 32, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar

        
class VAE_try(nn.Module):
    def __init__(self, image_channels=3, h_dim=8*2*2, z_dim=4):
        super(VAE_try, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  #N, 16, 16, 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),   #N, 32, 8, 8
            nn.ReLU(),
            nn.Conv2d(32, 16, 8, stride=2, padding=1),    #N, 64, 2, 2
            nn.ReLU(),
            nn.Conv2d(16, 8, 2, stride=2, padding=1),    #N, 64, 1, 1
            nn.Flatten(1,-1),
            #nn.Linear(8*2*2, z_dim)

        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            
            #nn.Linear(z_dim, 8*2*2),
            nn.Unflatten(1, (8, 2, 2)),
            nn.ConvTranspose2d(8, 16, 2, stride=2, padding=1),  #N, 32, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 8, stride=2, padding=1),  #N, 32, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   #N, 16, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   #N, 1, 32, 32
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar


class VAE_mlp(nn.Module):
    def __init__(self, image_size=32*32, h_dim=100, z_dim=4):
        super(VAE_mlp, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),  #input layer
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),   #h1
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),    #h1
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),    #h1
            nn.ReLU()
            #nn.Linear(100,z_dim)  # latent layer
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            #nn.Linear(z_dim, h_dim),  #input layer
            #nn.ReLU(),
            nn.Linear(h_dim, h_dim),   #h1
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),    #h1
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),    #h1
            nn.ReLU(),
            nn.Linear(h_dim, 32*32),  # latent layer
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar


class VAE_try_MRI(nn.Module):
    def __init__(self, image_channels=3, h_dim=16*9*9, z_dim=4):
        super(VAE_try_MRI, self).__init__()
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
            nn.Flatten(1,-1)

        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            
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
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar


class VAE_mlp_MRI(nn.Module):
    def __init__(self, image_size=96*96, h_dim=1000, z_dim=4):
        super(VAE_mlp_MRI, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, 1000),  #input layer
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
            nn.ReLU()
            #nn.Linear(100,z_dim)  # latent layer
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            #nn.Linear(z_dim, h_dim),  #input layer
            #nn.ReLU(),
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
            nn.Linear(1000, image_size),  # latent layer
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar

class ConvVAE_circle(nn.Module):
    def __init__(self, image_channels=1, h_dim=5*4, z_dim=2):
        super(ConvVAE_circle, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 5, 3, stride=2, padding=1),  #N, 16, 16, 16
            Sin(),
            nn.Conv1d(5, 5, 3, stride=2, padding=1),   #N, 32, 8, 8
            Sin(),
            nn.Conv1d(5, 5, 3, stride=1, padding=1),   #N, 32, 8, 8
            Sin(),            
            nn.Flatten(1,-1),
            #nn.Linear(8*2*2, z_dim)

        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            
            #nn.Linear(z_dim, 8*2*2),
            nn.Unflatten(1, (5, 4)),
            nn.ConvTranspose1d(5, 5, 3, stride=2, padding=1, output_padding=1),  #N, 32, 8, 8
            Sin(),
            nn.ConvTranspose1d(5, 5, 3, stride=1, padding=1, output_padding=0),  #N, 32, 8, 8
            Sin(),
            nn.ConvTranspose1d(5, 1, 3, stride=2, padding=1, output_padding=0),  #N, 32, 8, 8
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar


class ConvVAE_circle1024(nn.Module):
    def __init__(self, image_channels=1, h_dim=5*4, z_dim=2):
        super(ConvVAE_circle1024, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 5, 3, stride=2, padding=1),  #N, 16, 16, 16
            Sin(),
            nn.Conv1d(5, 5, 3, stride=2, padding=1),   #N, 32, 8, 8
            Sin(),
            nn.Conv1d(5, 1, 3, stride=1, padding=1),   #N, 32, 8, 8
            Sin(),            
            nn.Flatten(1,-1),
            #nn.Linear(8*2*2, z_dim)

        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            
            #nn.Linear(z_dim, 8*2*2),
            nn.Unflatten(1, (1, h_dim)),
            nn.ConvTranspose1d(1, 5, 3, stride=2, padding=1, output_padding=1),  #N, 32, 8, 8
            Sin(),
            nn.ConvTranspose1d(5, 5, 3, stride=1, padding=1, output_padding=0),  #N, 32, 8, 8
            Sin(),
            nn.ConvTranspose1d(5, 1, 3, stride=2, padding=1, output_padding=1),  #N, 32, 8, 8
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar


class VAE_mlp_circle_new(nn.Module):
    def __init__(self, image_size=15, h_dim=6, z_dim=2):
        super(VAE_mlp_circle_new, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),  #input layer
            Sin(),
            nn.Linear(h_dim, h_dim),   #h1
            Sin(),
            nn.Linear(h_dim, h_dim),    #h1
            Sin(),
            #nn.Linear(100,z_dim)  # latent layer
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            #nn.Linear(z_dim, h_dim),  #input layer
            #Sin(),
            nn.Linear(h_dim, h_dim),    #h1
            Sin(),
            nn.Linear(h_dim, h_dim),    #h1
            Sin(),
            nn.Linear(h_dim, image_size),  # latent layer
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar