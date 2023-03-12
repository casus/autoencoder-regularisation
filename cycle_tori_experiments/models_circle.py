import torch
import torch.nn as nn
from activations import Sin

class ConvEncoder(torch.nn.Module):
    def __init__(self, inp_size, no_filters, no_layers,
                 kernel_size, latent_dim, activation):
        super(ConvEncoder, self).__init__()
        
        self.no_layers = no_layers
        self.dim_x = inp_size
        self.inlayer = nn.Conv1d(in_channels=1, out_channels=no_filters,
                                 kernel_size=kernel_size, stride=2, padding=1)
        self.dim_x = int((self.dim_x - kernel_size + 2 * 1) / 2 + 1)
        
        self.convlayers = nn.ModuleList([])
        
        for i in range(0, self.no_layers):
            self.convlayers.append(nn.Conv1d(in_channels=no_filters, out_channels=no_filters,
                                             kernel_size=kernel_size, stride=2, padding=1))
            
            self.dim_x = int((self.dim_x - kernel_size + 2 * 1) / 2 + 1)

        self.linlayer = nn.Linear(int(self.dim_x  * no_filters), latent_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation((self.inlayer(x.unsqueeze(1))))
        for i in range(self.no_layers):
            x = self.activation(
                    self.convlayers[i](x)
                )
        x = self.activation(
                self.linlayer(torch.flatten(x, start_dim=1))
            )
        return x
    
class ConvDecoder(torch.nn.Module):
    def __init__(self, out_size, no_filters, no_layers,
                 kernel_size, latent_dim, dim_x, activation):
        super(ConvDecoder, self).__init__()
        
        self.no_layers = no_layers
        self.dim_x = dim_x   
        self.out_size = out_size
        self.no_filters = no_filters
        self.inlayer = nn.Linear(latent_dim, int(self.dim_x * no_filters))
        
        self.convlayers = nn.ModuleList([])
        
        for i in range(0, self.no_layers):
            self.convlayers.append(nn.ConvTranspose1d(in_channels=no_filters, out_channels=no_filters, kernel_size=kernel_size, stride=2, padding=1))
            
        self.outlayer = nn.Conv1d(in_channels=no_filters, out_channels=1,
                                 kernel_size=kernel_size, stride=2, padding=1)
        self.activation = activation
          
    def forward(self, x):
        x = self.activation((self.inlayer(x))).reshape(x.shape[0], self.no_filters, self.dim_x)
        for i in range(self.no_layers):
            x = self.activation(
                    self.convlayers[i](x)
                )
        x = torch.nn.functional.interpolate(self.outlayer(x), size=self.out_size, mode='linear')
        return x

class ConvAE(torch.nn.Module):
    def __init__(self, inp_size, no_filters, no_layers, kernel_size, latent_dim, activation):
        super(ConvAE, self).__init__()
        
        self.encoder = ConvEncoder(inp_size, no_filters, no_layers, kernel_size, latent_dim, activation)
        self.decoder = ConvDecoder(inp_size, no_filters, no_layers, kernel_size, latent_dim, self.encoder.dim_x, activation)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from models import MLP

class VAEDecoder(nn.Module):
    def __init__(self, out_size, no_filters, no_layers,
                 kernel_size, latent_dim, dim_x, activation=Sin()):
        super(VAEDecoder,self).__init__()
        self.decoder = ConvDecoder(out_size, no_filters, no_layers,
                 kernel_size, latent_dim, dim_x, activation=Sin())
    def forward(self, x):
        x = self.decoder.forward(x)
        return x

class VAEEncoder(nn.Module):
    def __init__(self, inp_size, no_filters, no_layers, kernel_size, 
                 latent_dim, activation=Sin()):
        super(VAEEncoder, self).__init__()
        self.activation = activation
        self.encoder = ConvEncoder(inp_size, no_filters, no_layers,
                 kernel_size, latent_dim, activation)
        self.fc_mu = nn.Linear(self.encoder.dim_x*no_filters, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.dim_x*no_filters, latent_dim)

    def forward(self, x):
        #x = self.activation((self.encoder.inlayer(x.unsqueeze(1))))
        x = self.activation((self.encoder.inlayer(x)))
        for i in range(self.encoder.no_layers):
            x = self.activation(
                    self.encoder.convlayers[i](x)
                )
        flatten = x.view(x.size(0), -1)
        mu, logvar = self.activation(self.fc_mu(flatten)), self.activation(self.fc_logvar(flatten))
        return mu, logvar
    
###################
### VAE
###################

class BetaVAE(nn.Module):
    def __init__(self, inp_size, no_filters, no_layers,
                  kernel_size, latent_dim, activation = Sin(), beta=1, use_mu=1):
        super(BetaVAE, self).__init__()

        
        self.use_mu = use_mu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._encoder = VAEEncoder(inp_size, no_filters, no_layers, kernel_size, 
                 latent_dim, activation=Sin()).to(device)
        self.decoder = VAEDecoder(inp_size, no_filters, no_layers,
                 kernel_size, latent_dim, self._encoder.encoder.dim_x, activation).to(device)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        
    def reparametrize(self, mu, logvar, training):
        if training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu

    def encoder(self, x):
        z, _, _ = self.encode(x)
        return z
        
    def encode(self, x, training=True):
        mu, logvar = self._encoder(x)
        z = self.reparametrize(mu, logvar,training)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder.forward(z)
        return z

    def forward(self, x):
        b, mu, logvar = self.encode(x)
        z = self.decode(b)
        
        '''
        if self.use_mu:
            p_ref = mu
        else:
            p_ref = b
        param = self.mlp(p_ref)
        return z, mu, logvar, param
        '''
        return z, mu, logvar

#both functions are from: https://github.com/1Konny/Beta-VAE/tree/977a1ece88e190dd8a556b4e2efb665f759d0772
def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0
    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        #x_recon = F.sigmoid(x_recon)
        #x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

from models import Encoder, Decoder
class VAEMLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, inp_dim, no_layers , activation=Sin()):
        super(VAEMLPDecoder,self).__init__()
        self.decoder = Decoder(latent_dim, hidden_size, inp_dim, no_layers, activation)
    def forward(self, x):
        x = self.decoder.forward(x)
        return x

class VAEMLPEncoder(nn.Module):
    def __init__(self, inp_dim, hidden_size, latent_dim, no_layers, activation=Sin()):
        super(VAEMLPEncoder, self).__init__()
        self.activation = activation
        self.encoder = Encoder(inp_dim, hidden_size, latent_dim, no_layers, activation)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        for layer in self.encoder.lin_layers[:-1]:
            x = self.activation(layer(x))
        flatten = x.view(x.size(0), -1)
        mu, logvar = self.activation(self.fc_mu(flatten)), self.activation(self.fc_logvar(flatten))
        return mu, logvar
    
###################
### VAE
###################

class MLPVAE(nn.Module):
    def __init__(self, inp_dim, hidden_size, latent_dim, no_layers, activation=Sin(), beta=1, use_mu=1):
        super(MLPVAE, self).__init__()
        self.use_mu = use_mu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._encoder = VAEMLPEncoder(inp_dim, hidden_size, latent_dim, no_layers, activation).to(self.device)
        self.decoder = VAEMLPDecoder(latent_dim, hidden_size, inp_dim, no_layers, activation).to(self.device)
        self.beta = beta

    def reparametrize(self, mu, logvar, training):
        if training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu

    def encoder(self, x):
        z, _, _ = self.encode(x)
        return z
    
    def encoder_(self, x):
        z, _, _ = self.encode(x, training=False)
        return z
        
    def encode(self, x, training=True):
        mu, logvar = self._encoder(x)
        z = self.reparametrize(mu, logvar, training)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder.forward(z)
        return z

    def forward(self, x):
        b, mu, logvar = self.encode(x)
        z = self.decode(b)
        
        '''
        if self.use_mu:
            p_ref = mu
        else:
            p_ref = b
        param = self.mlp(p_ref)
        return z, mu, logvar, param
        '''
        return z, mu, logvar
