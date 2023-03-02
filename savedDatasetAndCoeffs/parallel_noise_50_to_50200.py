import sys
sys.path.append('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders')

import concurrent.futures
import secrets
import time
from unittest import result

from pkg_resources import find_distributions


import numpy as np
import torch
import sys
#sys.path.insert(1, '/home/suarez08/PhD_PINNs/PIPS_framework')
from jmp_solver1.sobolev import Sobolev
from jmp_solver1.sobolev import Sobolev
from jmp_solver1.solver import Solver
from jmp_solver1.utils import matmul
import jmp_solver1.surrogates
import time
#sys.path.insert(1, '/home/suarez08/minterpy/src')
import minterpy as mp
from jmp_solver1.diffeomorphisms import hyper_rect
import matplotlib
import matplotlib.pyplot as plt
#style.use('dark_background')
matplotlib.rcdefaults() 
torch.set_printoptions(precision=10)
import nibabel as nib     # Read / write access to some common neuroimaging file formats
import ot
import scipy
import scipy.integrate

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')

torch.set_default_dtype(torch.float64)

#trainDataset = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/trainDataSet.pt').to(device)
testDataset = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/testDataSet.pt').to(device)


pert = torch.rand(testDataset.shape).to(device)

noise_order = torch.linspace(1e-3,100* 1e-1, 100)





deg_quad = 80
u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
x = np.linspace(-1,1,96)
X_p = u_ob.data_axes([x,x]).T


b = np.linspace(-1,1,96)#np.array([x[0]])#np.linspace(-1,1,100)
xf= np.linspace(-1,1,96)#x#np.linspace(-1,1,100)
X_test = u_ob.data_axes([b,xf]).T


def get_all_thetas(listedImage):
    Fr = torch.tensor(listedImage).reshape(96*96)

    def grad_x(t,theta):
        theta_t = torch.tensor(theta)
        return -2*torch.matmul(X_p.T,(torch.matmul(X_p,theta_t)-Fr)).detach().numpy()

    def give_theta_t():
        start = time.time()
        u_ob.set_weights_val(0.0)
        theta_0 =  list(u_ob.parameters())[0][0]
        dt = 0.01
        theta_t = theta_0
        for k in range(20):
            theta_int =  scipy.integrate.RK45(grad_x, 0.1, theta_t.detach().numpy(), 100)
            theta_int.step()
            theta_t = torch.tensor(theta_int.y)
        return theta_t

    act_theta = give_theta_t()
    return act_theta


start = time.perf_counter()

all_pert_coeffs = torch.tensor([])

all_coeffs = torch.tensor([])

noise_order = torch.linspace(1e-3,100* 1e-1, 100)

#testDataset =testDataset + 2e-1 *  torch.rand(testDataset.shape)
for i in range(len(noise_order)):

    #i = 0
    testDataset =testDataset + noise_order[i] *  torch.rand(testDataset.shape)

    testDataset = testDataset[:200]

    with concurrent.futures.ProcessPoolExecutor() as executor:

        trainDataset = testDataset.tolist()

        results = executor.map(get_all_thetas, trainDataset)
        
        for result in results:
            all_coeffs = torch.cat((all_coeffs, result))

        all_coeffs = all_coeffs.reshape(int(all_coeffs.shape[0]/ (deg_quad+1)**2),int((deg_quad+1)**2))
        all_coeffs = all_coeffs.unsqueeze(0)
        #torch.save(all_coeffs, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/exp_testCoeff_2noise_50_to_50200.pt')

        print('all_coeffs.shape',all_coeffs.shape)

    all_pert_coeffs = torch.cat((all_pert_coeffs, all_coeffs))
    all_coeffs = torch.tensor([])

torch.save(all_pert_coeffs, '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/savedDatasetAndCoeffs/testCoeffs50000To50200Noise0To10_.pt')

print('all_pert_coeffs.shape', all_pert_coeffs.shape)

finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} seconds')

