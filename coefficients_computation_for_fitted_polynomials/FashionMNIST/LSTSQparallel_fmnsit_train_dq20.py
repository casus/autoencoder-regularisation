import concurrent.futures
import secrets
import time
from unittest import result

from pkg_resources import find_distributions


import numpy as np
import torch
import sys
sys.path.append('./')

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
from datasets import  getDataset
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')

torch.set_default_dtype(torch.float64)


train_loader, test_loader, noChannels, dx, dy = getDataset('FashionMNIST', 60000, False)

trainDataset, train_labels = next(iter(train_loader))
testDataset, test_labels = next(iter(test_loader))

'''trainDataset = torch.load('/home/ramana44/withR_matrix_Fmnist_RK_method/coeffs_saved/trainImages.pt')
testDataset = torch.load('/home/ramana44/withR_matrix_Fmnist_RK_method/coeffs_saved/testImages.pt')'''


trainDataset = trainDataset[:50]
testDataset = testDataset[:50]
print('trainDataset.shape', trainDataset.shape)

deg_quad = 20
u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
x = np.linspace(-1,1,32)
X_p = u_ob.data_axes([x,x]).T


b = np.linspace(-1,1,32)#np.array([x[0]])#np.linspace(-1,1,100)
xf= np.linspace(-1,1,32)#x#np.linspace(-1,1,100)
X_test = u_ob.data_axes([b,xf]).T

start = time.perf_counter()

all_coeffs_lstsq = torch.tensor([])
for i in range(trainDataset.shape[0]):
    orig = trainDataset[i][0]
    get = np.linalg.lstsq(np.array(X_p), orig.reshape(32*32), rcond='warn')
    testRK = torch.tensor(get[0]).unsqueeze(0)
    #print(testRK.shape)
    all_coeffs_lstsq = torch.cat((all_coeffs_lstsq, testRK),0)

torch.save(all_coeffs_lstsq, './coefficients_computation_for_fitted_polynomials/FashionMNIST/saved_coefficients/LSTSQ_traincoeffs_FMNIST_dq'+str(deg_quad)+'.pt')

print('all_coeffs.shape',all_coeffs_lstsq.shape)

all_coeffs_lstsq = torch.tensor([])
for i in range(testDataset.shape[0]):
    orig = testDataset[i][0]
    get = np.linalg.lstsq(np.array(X_p), orig.reshape(32*32), rcond='warn')
    testRK = torch.tensor(get[0]).unsqueeze(0)
    #print(testRK.shape)
    all_coeffs_lstsq = torch.cat((all_coeffs_lstsq, testRK),0)

torch.save(all_coeffs_lstsq, './coefficients_computation_for_fitted_polynomials/FashionMNIST/saved_coefficients/LSTSQ_testcoeffs_FMNIST_dq'+str(deg_quad)+'.pt')


finish = time.perf_counter()
print('all_coeffs.shape',all_coeffs_lstsq.shape)
print(f'Finished in {round(finish-start, 2)} seconds')


