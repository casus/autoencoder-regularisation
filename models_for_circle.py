import torch
import torch.nn as nn
from activations import Sin

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class Autoencoder_linear(nn.Module):
    def __init__(self,latent_dim, dim):

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 6),  #input layer
            Sin(),
            nn.Linear(6, 6),    #h1
            Sin(),
            nn.Linear(6, 6),    #h1
            Sin(),
            nn.Linear(6,latent_dim),  # latent layer
            Sin()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 6),  #input layer
            Sin(),
            nn.Linear(6, 6),    #h1
            Sin(),
            nn.Linear(6, 6),    #h1
            Sin(),
            nn.Linear(6, dim),  # latent layer
            #nn.Sigmoid()
            Sin()
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
        
        #self.fc1 = nn.Linear(h_dim, z_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            Sin()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            Sin()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            Sin()
            )
        
        self.decoder = nn.Sequential(
            #nn.Linear(z_dim, h_dim),  #input layer
            #Sin(),
            nn.Linear(h_dim, h_dim),    #h1
            Sin(),
            nn.Linear(h_dim, h_dim),    #h1
            Sin(),
            nn.Linear(h_dim, image_size),  # latent layer
            #nn.Sigmoid()
            Sin()
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
        
        self.fc1 = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            Sin()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            Sin()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            Sin()
            )
        
        self.decoder = nn.Sequential(
            
            #nn.Linear(z_dim, 8*2*2),
            nn.Unflatten(1, (5, 4)),
            nn.ConvTranspose1d(5, 5, 3, stride=2, padding=1, output_padding=1),  #N, 32, 8, 8
            Sin(),
            nn.ConvTranspose1d(5, 5, 3, stride=1, padding=1, output_padding=0),  #N, 32, 8, 8
            Sin(),
            nn.ConvTranspose1d(5, 1, 3, stride=2, padding=1, output_padding=0),  #N, 32, 8, 8
            #nn.Sigmoid()
            Sin()
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

        
class ConvoAE_for_1024(nn.Module):
    def __init__(self, latent_dim):
        # 1 as input in Conv2d indicates the number of channels
        super().__init__()
        #N, 1, 32, 32
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 5, 3, stride=2, padding=1),  #N, 16, 16, 16
            Sin(),
            nn.Conv1d(5, 5, 3, stride=2, padding=1),   #N, 32, 8, 8
            Sin(),
            nn.Conv1d(5, 1, 3, stride=1, padding=1),   #N, 32, 8, 8
            Sin(),            
            nn.Flatten(1,-1),
            nn.Linear(256, latent_dim),
            Sin()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Unflatten(1, (1, 256)),
            nn.ConvTranspose1d(1, 5, 3, stride=2, padding=1, output_padding=1),  #N, 32, 8, 8
            Sin(),
            nn.ConvTranspose1d(5, 5, 3, stride=1, padding=1, output_padding=0),  #N, 32, 8, 8
            Sin(),
            nn.ConvTranspose1d(5, 1, 3, stride=2, padding=1, output_padding=1),  #N, 32, 8, 8
            Sin()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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
        
        self.fc1 = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            Sin()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(h_dim, z_dim)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(z_dim, h_dim)
            )
        
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