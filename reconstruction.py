import torch
import numpy as np
from scipy import interpolate

from quadrature import getQuad
from quadrature import transform

def bilinear_interp(img, x, y):
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor

    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, img.shape[1]-1)
    x1 = torch.clamp(x1, 0, img.shape[1]-1)
    y0 = torch.clamp(y0, 0, img.shape[0]-1)
    y1 = torch.clamp(y1, 0, img.shape[0]-1)
    
    Ia = img[ y0, x0 ][0]
    Ib = img[ y1, x0 ][0]
    Ic = img[ y0, x1 ][0]
    Id = img[ y1, x1 ][0]
    
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))

    return torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)

def reco_loss(reconstruction, batch_x, points, weights, device):
    err_sqr = torch.square(torch.sub(reconstruction, batch_x))
    res = torch.sum(torch.sum(err_sqr, 0), 0)
    
    loss = torch.FloatTensor([0.]).to(device)

    for pt, wt in zip(points, weights):
        x = torch.reshape(pt[0], (1,))
        y = torch.reshape(pt[1], (1,))
        loss += bilinear_interp(res, x, y)*wt
    
    #print(loss/torch.sum(weights))
        
    return loss/torch.sum(weights)