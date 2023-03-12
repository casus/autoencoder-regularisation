import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def contractive_loss_function(W, x, recons_x, h, lam):
    #mse_loss = nn.BCELoss(size_average = False)
    #mse_loss = torch.nn.BCELoss(reduction='sum')
    mseLoss_nn = torch.nn.MSELoss()
    
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
    #mse = mse_loss(recons_x, x)
    mse = mseLoss_nn(recons_x, x)

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



def loss_fn_mlp_vae(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x.float(), x.float(), size_average=False)
    #BCE = F.mse_loss(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, reduction='sum')


    #BCE = torch.nn.MSELoss()(x, recon_x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def loss_fn_cnn_vae(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x.float(), x.float(), size_average=False)
    #BCE = F.mse_loss(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    #BCE = torch.nn.MSELoss()(x, recon_x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD