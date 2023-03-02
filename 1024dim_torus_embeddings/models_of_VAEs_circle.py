import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import nn
import os, os.path
from activations import Sin

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden, activation = F.relu):
        super(MLP, self).__init__()
        self.linear_layers = nn.ModuleList()
        self.activation = activation
        self.init_layers(input_size, output_size, hidden_size,num_hidden)
    
    
    def init_layers(self, input_size, output_size, hidden_size, num_hidden):
        self.linear_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden):
            self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
        self.linear_layers.append(nn.Linear(hidden_size, output_size))

        for m in self.linear_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        for i in range(len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            x = self.activation(x)
        x = self.linear_layers[-1](x)
        return x


class Encoder(torch.nn.Module):
    """Takes an image and produces a latent vector."""
    def __init__(self, inp_dim, hidden_size, latent_dim, no_layers = 3, activation = F.relu):
        super(Encoder, self).__init__()
        self.activation = activation

        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(np.prod(inp_dim), hidden_size))
        for i in range(no_layers):
            self.lin_layers.append(nn.Linear(hidden_size, hidden_size))
        self.lin_layers.append(nn.Linear(hidden_size, latent_dim))

        for m in self.lin_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        #x = x.view(x.size(0), -1)
        if len(x.shape)==3:
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        else:
            x = x.view(x.size(0), -1)
        for layer in self.lin_layers:
            x = self.activation(layer(x))
        #x = torch.tanh(self.lin_layers[-1](x))
        return x


class Decoder(torch.nn.Module):
    """ Takes a latent vector and produces an image."""
    def __init__(self, latent_dim, hidden_size, inp_dim, no_layers = 3, activation = F.relu):
        super(Decoder, self).__init__()
        self.activation = activation
        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(latent_dim, hidden_size))
        for i in range(no_layers):
            self.lin_layers.append(nn.Linear(hidden_size, hidden_size))
        self.lin_layers.append(nn.Linear(hidden_size, np.prod(inp_dim)))

        for m in self.lin_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.lin_layers[:-1]:
            x = self.activation(layer(x))
        x = self.lin_layers[-1](x)
        #x = torch.sigmoid(x) # squash into [0,1]
        return x


class AE(torch.nn.Module):
    def __init__(self, inp_dim, hidden_size, latent_dim, no_layers, activation):
        super(AE, self).__init__()
        self.encoder = Encoder(inp_dim, hidden_size, latent_dim, no_layers, activation)
        self.decoder = Decoder(latent_dim, hidden_size, inp_dim, no_layers, activation)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

############################  
#### ConvAE is starting here
############################

class ConvEncoder(torch.nn.Module):
    def __init__(self, inp_size, noChannels, channel_mult, no_layers,
                 kernel_size, latent_dim, activation, batchNorm=False):
        super(ConvEncoder, self).__init__()
        
        self.no_layers = no_layers
        self.dim_x = inp_size[0]
        self.dim_y = inp_size[1]
        self.batchNorm = batchNorm
        self.inlayer = nn.Conv2d(in_channels=noChannels, out_channels=channel_mult*1,
                                 kernel_size=(kernel_size, kernel_size), stride=2, padding=1)
        if self.batchNorm:
            self.normlayers = nn.ModuleList([])
            self.normlayers.append(nn.BatchNorm2d(channel_mult*1))
        self.dim_x = int((self.dim_x - kernel_size + 2 * 1) / 2 + 1)
        self.dim_y = int((self.dim_y - kernel_size + 2 * 1) / 2 + 1)
        
        self.convlayers = nn.ModuleList([])
        
        for i in range(1, self.no_layers):
            self.convlayers.append(nn.Conv2d(in_channels=channel_mult*(2**(i-1)), out_channels=channel_mult*(2**i), kernel_size=(kernel_size, kernel_size), stride=2, padding=1))
            
            self.dim_x = int((self.dim_x - kernel_size + 2 * 1) / 2 + 1)
            self.dim_y = int((self.dim_y - kernel_size + 2 * 1) / 2 + 1)
            
            if self.batchNorm:
                self.normlayers.append(nn.BatchNorm2d(channel_mult*(2**i)))
        self.linlayer = nn.Linear(int(self.dim_x * self.dim_y * 
                                      channel_mult * (2**(self.no_layers-1))), latent_dim)
        if batchNorm:
            self.normlayers.append(nn.BatchNorm1d(latent_dim))
        self.activation = activation

    def forward(self, x):
        if self.batchNorm:
            x = self.activation(self.normlayers[0](self.inlayer(x)))
            for i in range(self.no_layers-1):
                x = self.activation(
                    self.normlayers[i+1](
                        self.convlayers[i](x)
                    ))
            x = self.activation(
            #x = torch.tanh(
                self.normlayers[-1](
                    self.linlayer(torch.flatten(x, start_dim=1))
                ))
        else:
            x = self.activation((self.inlayer(x)))
            for i in range(self.no_layers-1):
                x = self.activation(
                        self.convlayers[i](x)
                    )
            x = self.activation(
            #x = torch.tanh(
                    self.linlayer(torch.flatten(x, start_dim=1))
                )
        return x
    
class ConvDecoder(torch.nn.Module):
    def __init__(self, out_size, noChannels, channel_mult, no_layers,
                 kernel_size, latent_dim, activation, batchNorm=False):
        super(ConvDecoder, self).__init__()
        
        self.no_layers = no_layers
        self.kernel_size = kernel_size
        self.channel_mult = channel_mult
        self.out_size = out_size
        self.get_insize()
        self.batchNorm = batchNorm
        self.linlayer = nn.Linear(latent_dim, int(self.dim_x * self.dim_y * 
                                      channel_mult * (2**(self.no_layers-1))))
        
        self.convlayers = nn.ModuleList([])
        self.convlayers.append(nn.ConvTranspose2d(channel_mult*(2**(no_layers-1)),
                                                  channel_mult*(2**(no_layers-2)),
                                 kernel_size=(kernel_size, kernel_size), stride=1, padding=1, bias=False))

        if batchNorm:
            self.normlayers = nn.ModuleList([])
            
            self.normlayers.append(nn.BatchNorm1d(int(self.dim_x * self.dim_y * 
                                      channel_mult * (2**(self.no_layers-1)))))
            self.normlayers.append(nn.BatchNorm2d(channel_mult*(2**(no_layers-2))))
  
        for i in range(self.no_layers-2, 0, -1):
            self.convlayers.append(nn.ConvTranspose2d(channel_mult*(2**i), channel_mult*(2**(i-1)), kernel_size=(kernel_size, kernel_size), stride=2, padding=1, bias=False))

            if self.batchNorm:
                self.normlayers.append(nn.BatchNorm2d(channel_mult*(2**(i-1))))  
                               
        self.convlayers.append(nn.ConvTranspose2d(channel_mult, noChannels, kernel_size=(kernel_size, kernel_size), stride=1, padding=1))

        self.activation = activation

    def get_insize(self):
        self.dim_x = self.out_size[0]
        self.dim_y = self.out_size[1]

        self.dim_x = int((self.dim_x - self.kernel_size + 2 * 1) / 2 + 1)
        self.dim_y = int((self.dim_y - self.kernel_size + 2 * 1) / 2 + 1)
        
        for i in range(1, self.no_layers):
            self.dim_x = int((self.dim_x - self.kernel_size + 2 * 1) / 2 + 1)
            self.dim_y = int((self.dim_y - self.kernel_size + 2 * 1) / 2 + 1)
          
    def forward(self, x):
        self.get_insize()
        if self.batchNorm:
            x = self.activation(self.normlayers[0](self.linlayer(x)))
            x = x.reshape(x.size()[0], self.channel_mult*(2**(self.no_layers-1)), self.dim_x, self.dim_y)
            for i in range(self.no_layers-1):
                x = self.activation(
                    self.normlayers[i+1](
                        self.convlayers[i](x)
                    ))
                self.dim_x = self.kernel_size - 2 * 1 + 1 * (self.dim_x - 1)
                self.dim_y = self.kernel_size - 2 * 1 + 1 * (self.dim_y - 1)
                x = torch.nn.functional.interpolate(x, size=(self.dim_x*2, self.dim_y*2), mode='bilinear')
        else:
            x = self.activation(self.linlayer(x))
            x = x.reshape(x.size()[0], self.channel_mult*(2**(self.no_layers-1)), self.dim_x, self.dim_y)
            for i in range(self.no_layers-1):
                x = self.activation(
                        self.convlayers[i](x)
                    )
                self.dim_x = self.kernel_size - 2 * 1 + 1 * (self.dim_x - 1)
                self.dim_y = self.kernel_size - 2 * 1 + 1 * (self.dim_y - 1)
                x = torch.nn.functional.interpolate(x, size=(self.dim_x*2, self.dim_y*2), mode='bilinear')
        #x = torch.sigmoid(
        #    self.convlayers[-1](x)
        #    )
        x = self.convlayers[-1](x)
        x = torch.nn.functional.interpolate(x, size=(self.out_size[0], self.out_size[1]), mode='bilinear')
        return x        

class ConvAE(torch.nn.Module):
    def __init__(self, inp_size, noChannels, channel_mult, no_layers, kernel_size, latent_dim, activation, batchNorm):
        super(ConvAE, self).__init__()
        
        self.encoder = ConvEncoder(inp_size, noChannels, channel_mult, no_layers, kernel_size, latent_dim, activation, batchNorm)
        self.decoder = ConvDecoder(inp_size, noChannels, channel_mult, no_layers, kernel_size, latent_dim, activation, batchNorm)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VAEDecoder(nn.Module):
    def __init__(self, out_size, noChannels, no_filters, no_layers,
                 kernel_size, latent_dim, activation=Sin()):
        super(VAEDecoder,self).__init__()
        self.decoder = ConvDecoder(out_size, noChannels, no_filters, no_layers,
                 kernel_size, latent_dim, activation=Sin())
    def forward(self, x):
        x = self.decoder.forward(x)
        return x

class VAEEncoder(nn.Module):
    def __init__(self, inp_size, noChannels, no_filters, no_layers, kernel_size, 
                 latent_dim, activation=Sin()):
        super(VAEEncoder, self).__init__()
        self.activation = activation
        self.encoder = ConvEncoder(inp_size, noChannels, no_filters, no_layers,
                 kernel_size, latent_dim, activation)
        self.fc_mu = nn.Linear(int(self.encoder.dim_x * self.encoder.dim_y * 
                                      no_filters * (2**(no_layers-1))), latent_dim)
        self.fc_logvar = nn.Linear(int(self.encoder.dim_x * self.encoder.dim_y * 
                                      no_filters * (2**(no_layers-1))), latent_dim)

    def forward(self, x):
        #print('x.shape', x.shape)
        x = self.activation((self.encoder.inlayer(x)))
        for i in range(self.encoder.no_layers-1):
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
    def __init__(self, inp_size, noChannels, no_filters, no_layers,
                  kernel_size, latent_dim, activation = Sin(), beta=1, use_mu=1):
        super(BetaVAE, self).__init__()        
        self.use_mu = use_mu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._encoder = VAEEncoder(inp_size, noChannels, no_filters, no_layers, kernel_size, 
                 latent_dim, activation=Sin()).to(device)
        self.decoder = VAEDecoder(inp_size, noChannels, no_filters, no_layers,
                 kernel_size, latent_dim, activation).to(device)

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
        #print('x.shape', x.shape)
        #x = x.reshape(x.shape[0], x.shape[2]*x.shape[3])
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        for layer in self.encoder.lin_layers[:-1]:
            x = self.activation(layer(x))
        flatten = x.view(x.size(0), -1)
        mu, logvar = self.activation(self.fc_mu(flatten)), self.activation(self.fc_logvar(flatten))
        return mu, logvar
    
###################
### VAE
###################

class MLPVAE(nn.Module):
    def __init__(self, inp_dim, hidden_size, latent_dim, no_layers, activation=Sin(), beta=1., use_mu=1):
        super(MLPVAE, self).__init__()
        self.use_mu = use_mu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._encoder = VAEMLPEncoder(inp_dim, hidden_size, latent_dim, no_layers, activation).to(self.device)
        self.decoder = VAEMLPDecoder(latent_dim, hidden_size, inp_dim, no_layers, activation).to(self.device)
        self.beta = beta

    def reparametrize(self, mu, logvar, training=True):
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