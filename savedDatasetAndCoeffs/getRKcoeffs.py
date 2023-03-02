import numpy as np
import torch
import sys
sys.path.insert(1, '/home/suarez08/PhD_PINNs/PIPS_framework')
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

trainDataset = torch.load('/home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/trainDataSet.pt')
testDataset = torch.load('/home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/testDataSet.pt')

#things common to all images first


#embedding step
deg_quad = 80
u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
x = np.linspace(-1,1,96)
X_p = u_ob.data_axes([x,x]).T


b = np.linspace(-1,1,96)#np.array([x[0]])#np.linspace(-1,1,100)
xf= np.linspace(-1,1,96)#x#np.linspace(-1,1,100)
X_test = u_ob.data_axes([b,xf]).T


trainDataset = trainDataset.tolist()

print('len(trainDataset)',len(trainDataset))

#things specific to the image
all_coeffs = torch.tensor([])
for i in range(3):

    #Fr = torch.tensor(trainDataset[i][0]).reshape(96*96)
    #Fr = torch.tensor(Fr).to('cpu')
    listedImage = trainDataset[i][0]

    def get_all_thetas(listedImage):
        Fr = torch.tensor(listedImage).reshape(96*96)
        Fr = torch.tensor(Fr).to('cpu')

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
                #u_ob.set_weights(theta_t)
                if (k + 1) % 5 == 0:
                    print(k)
            return theta_t

        act_theta = give_theta_t()
        return act_theta

    theta_ex = get_all_thetas(listedImage)

    all_coeffs = torch.cat((all_coeffs, theta_ex))

all_coeffs = all_coeffs.reshape(int(all_coeffs.shape[0]/ (deg_quad+1)**2),int((deg_quad+1)**2))

print('shape of all_coeffs : ', all_coeffs.shape)

#torch.save(theta_ex, '/home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/theta_tExample.pt')

print('theta saved')

im = torch.matmul(X_p, all_coeffs[2])
im[torch.where(im<0)] = 0
im = im.reshape(96,96)

plt.imshow(im , origin='lower')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('/home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/ex_image')
plt.show()
plt.close()

print('image saved')


plt.imshow(Fr.reshape(96,96) , origin='lower')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('/home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/actual_image')
plt.show()
plt.close()

print('actual image saved')