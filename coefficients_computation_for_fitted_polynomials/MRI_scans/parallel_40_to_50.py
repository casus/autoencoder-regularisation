import sys
sys.path.append('./')
import concurrent.futures
import secrets
import time
from unittest import result

from pkg_resources import find_distributions


import numpy as np
import torch
import sys
from jmp_solver1.sobolev import Sobolev
from jmp_solver1.sobolev import Sobolev
from jmp_solver1.solver import Solver
from jmp_solver1.utils import matmul
import jmp_solver1.surrogates
import time
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

trainDataset = torch.load('/home/ramana44/oasis_mri_2d_slices_hybridAutoencodingAlphaHalf_enc_81_RK_here/savedDatasetAndCoeffs/trainDataSet.pt')
#testDataset = torch.load('/home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/testDataSet.pt')


trainDataset = trainDataset[40000:]


deg_quad = 70
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


all_coeffs = torch.tensor([])

with concurrent.futures.ProcessPoolExecutor() as executor:

    trainDataset = trainDataset.tolist()

    results = executor.map(get_all_thetas, trainDataset)
    
    for result in results:
        all_coeffs = torch.cat((all_coeffs, result))

    all_coeffs = all_coeffs.reshape(int(all_coeffs.shape[0]/ (deg_quad+1)**2),int((deg_quad+1)**2))
    torch.save(all_coeffs, './coefficients_computation_for_fitted_polynomials/MRI_scans/saved_coefficients/LSTQScoeff_40_to_50.pt')

    print('all_coeffs.shape',all_coeffs.shape)
finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} seconds')