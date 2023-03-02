import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import src.minterpy.transformation as Transform
import matplotlib.pyplot as plt
import derivativ_utils as u


# This Network learns a given image pixel by pixel
class PolynomialNeuralSolver(nn.Module):

    def __init__(self, lb, ub, noLayers=4, noFeatures=300, activation=torch.tanh, device="cpu", polynomialDegree=2,lp=2):
        """
        This function creates the components of the Neural Network
        :param no_layers: Number of layers used
        :param no_features: Number of features used
        :param activation: Activation function used
        """
        torch.manual_seed(2342)
        super(PolynomialNeuralSolver, self).__init__()
        self.no_layers = noLayers
        self.no_features = noFeatures
        self.lin_layers = nn.ModuleList()
        self.activation = activation
        self.lb = torch.Tensor(lb).float().to(device)
        self.ub = torch.Tensor(ub).float().to(device)
        self.device = device
        self.init_layers()
        
        ### initialisation of our polynomial
        self.one = torch.tensor([1]).float()
        M = 2 # inputDimension
        input_para = (M,polynomialDegree,lp)
        self.test_tr = Transform.Transformer(*input_para)
        self.PP = torch.from_numpy(self.test_tr.tree.grid_points).t().float()
        self.gamma = self.test_tr.exponents
        self.set_device(device)

    def init_layers(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """

        self.lin_layers.append(nn.Linear(2, self.no_features))
        for i in range(self.no_layers):
            inFeatures = self.no_features
            self.lin_layers.append(nn.Linear(self.no_features, self.no_features))
        self.lin_layers.append(nn.Linear(self.no_features, 1))

        for m in self.lin_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward_polynomial(self):
        x = self.PP
        for i in range(0, len(self.lin_layers) - 1):
            x = self.lin_layers[i](x)
            x = self.activation(x)
        x = self.lin_layers[-1](x)
        return x
    
    
    def canon_basis(self, x):
        res = torch.stack([torch.ones(x.shape[0]), x[:,0], x[:,0] ** 2, x[:,1], x[:,0] * x[:,1], x[:,1]**2])
        return res
    
    def canon_basis_dx0(self, x):
        res = torch.stack([torch.zeros(x.shape[0]), torch.ones(x.shape[0]), 2 * x[:,0], torch.zeros(x.shape[0]), x[:,1], torch.zeros(x.shape[0])])
        return res
    
    
    def canon_basis_dx1(self, x):
        res = torch.stack([torch.zeros(x.shape[0]), torch.zeros(x.shape[0]), torch.zeros(x.shape[0]), torch.ones(x.shape[0]), x[:,1], x[:,0]])
        return res
    
    
    def compute_1st_derivatives(self, x_in):
        # compute lagrange coefficients
        lagrange_coefs = self.forward_polynomial()

        # convert into canonical basis
        canon_coefs = self.test_tr.transform_l2c(lagrange_coefs).view(1,-1)

        # evaluate polynomial in asked positions x_in
        x_in = 2.0 * (x_in - self.lb) / (self.ub - self.lb) - 1.0
        
        # first derivative in x_0 direction
        gamma_dx0, constants_dx0 = self.test_tr.getGammaDXi(i = 0)
        # first derivative in x_1 direction
        gamma_dx1, constants_dx1 = self.test_tr.getGammaDXi(i = 1)
        # predict values
        Y_hat_dx0 = torch.mm( (canon_coefs*constants_dx0).float(), self.test_tr.buildCanonicalVandermonde(x_in, gamma_dx0).t())
        Y_hat_dx1 = torch.mm( (canon_coefs*constants_dx1).float(), self.test_tr.buildCanonicalVandermonde(x_in, gamma_dx1).t())
        
        return Y_hat_dx0, Y_hat_dx1
    
    
    def forward(self, x_in):
        # compute lagrange coefficients
        lagrange_coefs = self.forward_polynomial()
        
        # convert into canonical basis
        canon_coefs = self.test_tr.transform_l2c(lagrange_coefs).view(1,-1)
        
        # evaluate polynomial in asked positions x_in
        x_in = 2.0 * (x_in - self.lb) / (self.ub - self.lb) - 1.0
        noElem, _ = x_in.shape
        
        Y_hat = torch.mm(canon_coefs, self.test_tr.buildCanonicalVandermonde(x_in, self.gamma).t())
        
        return Y_hat
        
    
    def set_device(self, device):
        self.device = device
        if (str(device) == "cpu"):
            self.dtype = torch.FloatTensor
        else:
            self.dtype = torch.cuda.FloatTensor
    

    # Outputs the first and second derivatives of the learned image using Automatic Differentiation
    def net(self, x, y):
        """
        Function that calculates the nn output at postion x
        :param x: position
        :return: Solutions and their gradients
        """
        dim = x.shape[0]
        x = Variable(x, requires_grad=True)
        y = Variable(y, requires_grad=True)
        X = torch.cat([x, y], 1).type(self.dtype)
        Ex = torch.squeeze(self.forward(X)).type(self.dtype)
        grads = torch.ones([dim]) if str(self.device) =="cpu" else torch.ones([dim]).cuda()
        Ex_pds = torch.autograd.grad(Ex, [x, y], create_graph=True, grad_outputs=grads)
        Ex_x = Ex_pds[0].reshape([dim])
        Ex_y = Ex_pds[1].reshape([dim])

        # compute partial derivatives
        Ex_xx = torch.autograd.grad(Ex_x, x, create_graph=True, grad_outputs=grads)[0]
        Ex_yy = torch.autograd.grad(Ex_y, y, create_graph=False, grad_outputs=grads)[0]

        return Ex, Ex_x, Ex_y, Ex_xx, Ex_yy
    
    def getName(self):
        return "PNS_%d_%d" % (self.no_layers, self.no_features)

    
    # Used for LBFGS Optimization in AutoDiffNet
    def closure(self):
        out = self(self.coord)
        
        Ex_2d = out.view((1, 1, 150, 150))
        gt_2d = self.gt.view((1, 1, 150, 150))
        loss = self.criterion(Ex_2d, gt_2d)
        self.optim.zero_grad()
        loss.backward()
        return loss
    
    
    # Train AutoDiffNet using LBFGS or Adam
    def train(self, gt, optimizer='LBFGS', lr=1e-4, epochs=100):
        # Initialize input data for AutoDiffNet
        self.coord = gt.getCoords().to(self.device)
        self.gt = torch.from_numpy(gt.base(self.coord[:, 0].cpu().numpy(), self.coord[:, 1].cpu().numpy())).to(self.device)
        
        # Initialize loss function
        self.criterion = nn.MSELoss()

        # Training of PolynomialPINN using LBFGS or ADAM        
        if optimizer == 'LBFGS':
            self.optim = optim.LBFGS(self.parameters(), lr)
        elif optimizer == 'ADAM':
            self.optim = optimizer = optim.Adam(self.parameters(), lr)
            
        for epoch in range(epochs):
            self.optim.step(self.closure)
            # Show current progress in epochs
            if((epoch % 10) == 0):
                print("[%d] %.5f" % (epoch, self.closure()))
                res = self(self.coord)
                plt.imshow(res.view(150,150).detach().cpu().numpy())
          
        print("[%d] %.5f" % (epoch, self.closure()))
        res = self(gt.getCoords())
        plt.imshow(res.view(150,150).detach().cpu().numpy())
            
            
if __name__ == '__main__':
    # static parameters
    nx = 200
    ny = 200
    nt = 1000
    xmin = -10
    xmax = 10
    ymin = -10
    ymax = 10
    dt = 0.001
    numOfEnergySamplingPointsX = 100
    numOfEnergySamplingPointsY = 100
    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx": nx , "ny": ny, "nt": nt, "dt": dt}

    
    # Load data for t0
    lb = np.array([coordinateSystem["x_lb"], coordinateSystem["y_lb"], 0.])
    ub = np.array([coordinateSystem["x_ub"], coordinateSystem["y_ub"], coordinateSystem["nt"] * coordinateSystem["dt"]])
    
    PPINN = PolynomialPINN(ub,lb,noFeatures = 100)
    
    gt = u.Schrodinger2DGroundTruth(scalingFactor = 0.1)
    
    gt_base = torch.from_numpy(gt.base(gt.coord[:, 0].cpu().numpy(), gt.coord[:, 1].cpu().numpy())).to(device)
    
    coord_in = gt.coord
    
    gt_base.shape