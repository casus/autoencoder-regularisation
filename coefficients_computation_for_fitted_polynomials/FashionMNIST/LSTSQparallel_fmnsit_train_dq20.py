import concurrent.futures
import secrets
import time
from unittest import result

from pkg_resources import find_distributions


import numpy as np
import torch
import sys
sys.path.append('/home/ramana44/autoencoder-regularisation-')

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

trainDataset = torch.load('/home/ramana44/withR_matrix_Fmnist_RK_method/coeffs_saved/trainImages.pt')
testDataset = torch.load('/home/ramana44/withR_matrix_Fmnist_RK_method/coeffs_saved/testImages.pt')


#trainDataset = trainDataset[:100]
#testDataset = testDataset[:50]
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

torch.save(all_coeffs_lstsq, '/home/ramana44/withR_matrix_Fmnist_RK_method/coeffs_saved/LSTSQ_traincoeffs_FMNIST_dq'+str(deg_quad)+'.pt')

print('all_coeffs.shape',all_coeffs_lstsq.shape)

all_coeffs_lstsq = torch.tensor([])
for i in range(testDataset.shape[0]):
    orig = testDataset[i][0]
    get = np.linalg.lstsq(np.array(X_p), orig.reshape(32*32), rcond='warn')
    testRK = torch.tensor(get[0]).unsqueeze(0)
    #print(testRK.shape)
    all_coeffs_lstsq = torch.cat((all_coeffs_lstsq, testRK),0)

#torch.save(all_coeffs_lstsq, '/home/ramana44/withR_matrix_Fmnist_RK_method/coeffs_saved/LSTSQ_testcoeffs_FMNIST_dq'+str(deg_quad)+'.pt')


finish = time.perf_counter()
print('all_coeffs.shape',all_coeffs_lstsq.shape)
print(f'Finished in {round(finish-start, 2)} seconds')


#now loading the saved coeffs and printing

#print('now loading the saved coeffs and printing')

#traincoeffss = torch.load('/home/ramana44/withR_matrix_Fmnist_RK_method/coeffs_saved/LSTSQ_traincoeffs_FMNIST_dq'+str(deg_quad)+'.pt')
#testcoeffss = torch.load('/home/ramana44/withR_matrix_Fmnist_RK_method/coeffs_saved/LSTSQ_testcoeffs_FMNIST_dq'+str(deg_quad)+'.pt')

#print('traincoeffss.shape', traincoeffss.shape)

#print('testcoeffss.shape', testcoeffss.shape)