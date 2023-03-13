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

class AE_circ(nn.Module):
    def __init__(self,latent_dim):

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(15, 6),  #input layer
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
            nn.Linear(6, 15),  # latent layer
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#### DeepFluids AE
class Autoencoder_DeepFluid(nn.Module):
  
    # def __init__(self, noChannels, latent_dim, channel_mult=16, hidden_size = 1024, dx = 32):
    def __init__(self, n_channels, height = 32, width = 32, n_blocks=6, n_convlayers=3,  kernel_size=3, activate_bias=False, latent_size=1024):
        
        super().__init__()
        self.height = height
        self.width = width
        self.latent_size=latent_size
        self.act = nn.LeakyReLU(0.2)
        self.kernel_size = kernel_size
        self.activate_bias = activate_bias

        self.n_layers = n_blocks
        self.n_convlayers = n_convlayers
        self.n_channels = n_channels
        
        self.inlayer = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=(3, 3), padding=1)
        self.outlayer = nn.ConvTranspose2d(self.n_channels, self.n_channels, kernel_size=(3,3), padding=1)
        self.convlayerenc = nn.ModuleList([])
        self.convlayerdec = nn.ModuleList([])
        self.convredfil = nn.ModuleList([])
        self.convredfildec = nn.ModuleList([])
        self.downsample = nn.ModuleList([])
        self.linear_layers = nn.ModuleList([])
        self.pool = torch.nn.MaxPool2d((2,2))
        self.flintersize = self.get_flintersize()
        self.intersize_height = self.get_intersize(self.height)
        self.intersize_width = self.get_intersize(self.width)
        self.filters = 0
        self.init_layers()


    def get_intersize(self, inp_size, kernel_size, pad, stride):
        """
        This function calculates the reduction of dimension through convolution. 
        The calculation is performed only on a single dimension
        inp_size: size of the input dimension. 
        """
        size = inp_size
        for i in range(self.n_layers):
            size = int((size - kernel_size + 2 * pad) / stride + 1)
        return size
    
    def get_flintersize(self):
        """
        This function calculates the size of the latent space after the convolutional layer. 
        It uses the get_intersize function in order to calculate the dimensional reduction through convolution. 
        """
        size_x = self.get_intersize(self.height)
        size_y = self.get_intersize(self.width)
        latent_in_features = size_x * size_y * self.n_channels
        return latent_in_features
        
    def weights_init(self, m):
        """
        This function initializes the layers from a xavier distribution and its called with the apply function
        from the torch module class. 
        m: is the layer 
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
        
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.data, nn.init.calculate_gain('leaky_relu', 0.2)) #kaiming like initialisation
       
    def init_layers(self): 
        """
        Generates the layer following the block structure of the deep fluid paper.
        """
    
        print("Building linear layer for encoding latent space flatten %d <=> latent %d" % (self.flintersize, self.latent_size))
        self.linear_layers.append(nn.Linear(self.flintersize, self.flintersize, bias=self.activate_bias))
        self.linear_layers.append(nn.Linear(self.flintersize, self.latent_size, bias=self.activate_bias))
        self.linear_layers.append(nn.Linear(self.latent_size, self.flintersize, bias=self.activate_bias))
        self.linear_layers.append(nn.Linear(self.flintersize, self.flintersize, bias=self.activate_bias))
        self.lastconv = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=(self.kernel_size, self.kernel_size), padding=1)  
        if self.n_layers>1:
            for i in range(1, self.n_layers+1):
                self.downsample.append(nn.Conv2d(self.n_channels*2, self.n_channels, kernel_size=(self.kernel_size, self.kernel_size), stride=2, padding=1))
                self.convredfil.append(nn.Conv2d(self.n_channels, self.n_channels, kernel_size=(self.kernel_size, self.kernel_size), padding=1))
                self.convredfildec.append(nn.Conv2d(self.n_channels*2, self.n_channels, kernel_size=(self.kernel_size, self.kernel_size), padding=1))
        self.convredfil[-1] = nn.Conv2d(self.n_channels*2, self.n_channels, kernel_size=(self.kernel_size, self.kernel_size), padding=1)
        #self.convredfildec[-1] = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=(self.kernel_size, self.kernel_size), padding=1)
        if self.n_convlayers>1:
            for i in range(0, self.n_convlayers*self.n_layers):
                self.convlayerenc.append(nn.Conv2d(self.n_channels, self.n_channels, kernel_size=(self.kernel_size, self.kernel_size), padding=1))
                self.convlayerdec.append(nn.Conv2d(self.n_channels, self.n_channels, kernel_size=(self.kernel_size, self.kernel_size), padding=1))
                
        
    def encoder(self, x):
        """
        Performs the encoding step of the autoencoder
        returns the latent 
        x: input image 
        """
        x = self.inlayer(x)
        x = self.act(x)
        x0 = x
        for j in range(self.n_layers):          
            for i in range(1, self.n_convlayers):
                x = self.convlayerenc[j*self.n_convlayers + i](x)
                x = self.act(x) 
            x = torch.cat((x, x0), 1)

            if j < self.n_layers - 1:
                x = self.downsample[j+1](x)
                x = self.act(x)
                x0=x 

            #conv to reduce filters to self.n_channels of x  
            
            x = self.convredfil[j](x)
            x = self.act(x)
       
        self.filters = x.size()[1]
        flatten = torch.flatten(x, start_dim=1)

        x = self.linear_layers[0](flatten) # flatten => hidden
        x = self.act(x)
        x = self.linear_layers[1](x) # hidden => latent
        return x
    

    
    def decoder(self, x):
        """
        Performs the decoder step of the autoencoder.
        returns the reconstruction from the latent space. 
        x: latent spacs
        """
        x = self.linear_layers[2](x) # latent => hidden
        x = self.act(x)
        x = self.linear_layers[3](x) # hidden => flatten
        x = x.reshape(x.size()[0], self.filters, self.intersize_height, self.intersize_width)
        x0 = x
        for j in range(self.n_layers):
            for i in range(1, self.n_convlayers):
                x = self.convlayerdec[j*self.n_convlayers+i](x)
                x = self.act(x) 
            x = torch.cat((x, x0), 1)
            if j < self.n_layers - 1:
                x = torch.nn.functional.interpolate(x, size=(x.size()[2]*2,x.size()[3]*2), mode='bilinear')
                x = self.act(x)
            x = self.convredfildec[j](x)
            x = self.act(x)
            x0 = x
          
        x = self.lastconv(x)
        return x    
        
                        
    def forward(self, x):
        """
        Performs both the encoder and the decoder step 
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat    
    
    
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
        
        '''
        self.conv1 = nn.Conv2d(in_channels=noChannels, out_channels=channel_mult*1, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_mult*1, channel_mult*2, 4,2,1)
        self.norm2 = nn.BatchNorm2d(channel_mult*2)
        self.conv3 = nn.Conv2d(channel_mult*2, channel_mult*4,4,2,1)
        self.norm3 = nn.BatchNorm2d(channel_mult*4)
        self.conv4 = nn.Conv2d(channel_mult*4, channel_mult*8, 3,2,1)
        self.norm4 = nn.BatchNorm2d(channel_mult*8)

        self.conv_out_sz = self.norm4(self.conv4(self.norm3(self.conv3(self.norm2(self.conv2(self.conv1(torch.rand(1, noChannels, dx, dx)))))))).shape[2]
        self.fc1 = nn.Linear(self.conv_out_sz*self.conv_out_sz*channel_mult*8, latent_dim)
        self.norm_fc = nn.BatchNorm1d(latent_dim)
        
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.norm_fc(self.fc1(x)))
        return x
        '''

        
class ConvDecoder(torch.nn.Module):
    def __init__(self, in_size, out_size, noChannels, channel_mult, no_layers,
                 kernel_size, latent_dim, activation, batchNorm=False):
        super(ConvDecoder, self).__init__()
        
        self.no_layers = no_layers
        self.kernel_size = kernel_size
        self.channel_mult = channel_mult
        self.out_size = out_size
        #self.dim_x = in_size[0]
        #self.dim_y = in_size[1]
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
        x = torch.sigmoid(
            self.convlayers[-1](x)
            )
        x = torch.nn.functional.interpolate(x, size=(self.out_size[0], self.out_size[1]), mode='bilinear')
        return x        
'''
class ConvDecoder(torch.nn.Module):
    def __init__(self, noChannels, channel_mult, latent_dim, hidden_size):
        super(ConvDecoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, hidden_size)
        self.norm_fc1 = nn.BatchNorm1d(hidden_size)

        self.conv1 = nn.ConvTranspose2d(hidden_size, channel_mult*8, 6,1,1, bias=False)
        self.norm1 = nn.BatchNorm2d(channel_mult*8)
        self.conv2 = nn.ConvTranspose2d(channel_mult*8, channel_mult*4, 4,2,1, bias=False)
        self.norm2 = nn.BatchNorm2d(channel_mult*4)
        self.conv3 = nn.ConvTranspose2d(channel_mult*4, channel_mult*1, 4,2,1, bias=False)
        self.norm3 = nn.BatchNorm2d(channel_mult)
        self.conv4 = nn.ConvTranspose2d(channel_mult, noChannels, 4,2,1, bias=False)

    def forward(self, x):
        x = F.relu(self.norm_fc1(self.fc1(x)))
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = torch.sigmoid(self.conv4(x))
        return x
'''
class ConvAE(torch.nn.Module):
    def __init__(self, inp_size, noChannels, channel_mult, no_layers, kernel_size, latent_dim, activation, batchNorm):
        super(ConvAE, self).__init__()
        
        self.encoder = ConvEncoder(inp_size, noChannels, channel_mult, no_layers, kernel_size, latent_dim, activation, batchNorm)
        self.decoder = ConvDecoder([self.encoder.dim_x, self.encoder.dim_y], inp_size,
                                   noChannels, channel_mult, no_layers, kernel_size, latent_dim, activation, batchNorm)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
