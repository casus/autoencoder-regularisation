import numpy as np
import minterpy as mp
from minterpy.extras.regression import *
from matplotlib import pyplot as plt
import nibabel as nib     # Read / write access to some common neuroimaging file formats
import torch
torch.set_printoptions(precision=10)
from sklearn.neighbors import NearestNeighbors
import ot
import matplotlib
import matplotlib.pyplot as plt
#style.use('dark_background')
matplotlib.rcdefaults() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
from scipy import interpolate


#imports for sobolev fitting
import jmp_solver
#sys.path.insert(1, '/home/suarez08/PhD_PINNs/PIPS_framework')
from jmp_solver import *
from jmp_solver.sobolev import Sobolev
from jmp_solver.sobolev import Sobolev
from jmp_solver.solver import Solver
from jmp_solver.utils import matmul
import jmp_solver.surrogates
import time
#sys.path.insert(1, '/home/suarez08/minterpy/src')
import minterpy as mp
from jmp_solver.diffeomorphisms import hyper_rect


from get_data import get_data, get_data_train, get_data_val
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from datasets import InMemDataLoader
import torch.nn.functional as F
import torch
import nibabel as nib     # Read / write access to some common neuroimaging file formats
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#try out bilinear interpolation to get legendre image

def getLegendreImage_bilinearIngterpol(OriginalImage, originalAxis_x, originalAxis_y, legendreAxis_x, legendreAxis_y):

    f = interpolate.interp2d(originalAxis_x, originalAxis_y, OriginalImage, kind = 'linear')
    ImageFromInterpolatedFuncInLegendreGrid = f(legendreAxis_x, legendreAxis_y)

    return ImageFromInterpolatedFuncInLegendreGrid


#gradient sampling encoding method / smart grid method
def get_image_on_smartGrid(cleaned_image, ground_n, wall_n, originalAxis_x, originalAxis_y):

    ground_x = cleaned_image.mean(0)
    wall_y = cleaned_image.mean(1)
    ground_grad = abs(torch.gradient(ground_x)[0])
    wall_grad = abs(torch.gradient(wall_y)[0])
    SortgradVals_ground, SortgradInds_ground = torch.sort(ground_grad, 0)
    SortgradVals_wall, SortgradInds_wall = torch.sort(wall_grad, 0)
    selectedGradVals_ground = SortgradVals_ground[cleaned_image.shape[0]-ground_n:]
    selectedGradInds_ground = SortgradInds_ground[cleaned_image.shape[0]-ground_n:]
    selectedGradVals_wall = SortgradVals_wall[cleaned_image.shape[0]-wall_n:]
    selectedGradInds_wall = SortgradInds_wall[cleaned_image.shape[0]-wall_n:]
    ascen_ground_inds = selectedGradInds_ground.sort()
    ascen_wall_inds = selectedGradInds_wall.sort()
    #print('cleaned_image.shape',cleaned_image.shape)
    f = interpolate.interp2d(originalAxis_x, originalAxis_y, cleaned_image, kind = 'cubic')
    ImageFromInterpolatedFuncInIntelGrid = f(ascen_ground_inds[0], ascen_wall_inds[0])

    return ImageFromInterpolatedFuncInIntelGrid, ascen_ground_inds, ascen_wall_inds

#encoding the batch

def get_encoded_batch(batch_x, embeddedGridLength_x, embeddedGridLength_y, method):

    originalAxis_x = np.linspace(0, batch_x.shape[2]-1, batch_x.shape[2])
    originalAxis_y = np.linspace(0, batch_x.shape[3]-1, batch_x.shape[3])
    legendreGridX_length = embeddedGridLength_x
    legendreGridY_length = embeddedGridLength_y
    deg_x = legendreGridX_length
    deg_y = legendreGridY_length
    q_x, _ = np.polynomial.legendre.leggauss(deg_x)
    q_y, _ = np.polynomial.legendre.leggauss(deg_y)
    legendreAxis_x = batch_x.shape[2]/2 *(q_x+1)
    legendreAxis_y = batch_x.shape[2]/2 *(q_y+1)
    Q_exact_sorted_x = legendreAxis_x.reshape(-1,1)
    knn_x = NearestNeighbors(n_neighbors=2)
    knn_x.fit(Q_exact_sorted_x)
    Q_exact_sorted_y = legendreAxis_y.reshape(-1,1)
    knn_y = NearestNeighbors(n_neighbors=2)
    knn_y.fit(Q_exact_sorted_y)
    batch_Enc = torch.tensor([])
    for i in range(batch_x.shape[0]):
        if(method == 'grad_sampled'):
            imageEmbeddedcoded,_,_ = get_image_on_smartGrid(batch_x[i].squeeze(0).cpu(), embeddedGridLength_x, embeddedGridLength_y,originalAxis_x, originalAxis_y)
        else:
            imageEmbeddedcoded = getLegendreImage_bilinearIngterpol(batch_x[i].squeeze(0).cpu(), originalAxis_x, originalAxis_y, legendreAxis_x, legendreAxis_y)
        imageEmbeddedcoded = torch.tensor(imageEmbeddedcoded).unsqueeze(0)
        batch_Enc = torch.cat((batch_Enc,imageEmbeddedcoded.unsqueeze(0)))
    
    return batch_Enc


#Image decode using scipy

def getDecodingUsingScipy(LegendreImage, legendreAxis_x, legendreAxis_y, originalAxis_x, originalAxis_y):

    f = interpolate.interp2d(legendreAxis_x, legendreAxis_y, LegendreImage, kind = 'cubic')
    DecodedImage = f(originalAxis_x, originalAxis_y)

    return DecodedImage


#Get Image on to original Grid from Legendre grid

#Local bilinear interpolation method

#get back original image



def imageReconInUniformGrid_bilinearMethod(legendreAxis_x, legendreAxis_y, originalAxis_x, originalAxis_y, imageLegGrad_GradBilinearInterpol, knn_x, knn_y, Q_exact_sorted_x, Q_exact_sorted_y):

    recon_uniform_grid_ = torch.tensor([])
    for i in range(len(originalAxis_x)):
        for j in range(len(originalAxis_y)):
            x_co = i
            y_co = j

            neighbour_positions_xco = knn_x.kneighbors(np.array([[x_co]]), return_distance=False)
            neighbour_positions_yco = knn_y.kneighbors(np.array([[y_co]]), return_distance=False)

            neighbour_coordinates_xco = torch.tensor(Q_exact_sorted_x[neighbour_positions_xco[0]])
            neighbour_coordinates_yco = torch.tensor(Q_exact_sorted_y[neighbour_positions_yco[0]])

            neighbour_coordinates_xco_sorted = torch.sort(neighbour_coordinates_xco, 0)[0]
            neighbour_coordinates_yco_sorted = torch.sort(neighbour_coordinates_yco, 0)[0]

            x_co_1 = neighbour_coordinates_xco_sorted[0][0]
            x_co_2 = neighbour_coordinates_xco_sorted[1][0]

            y_co_1 = neighbour_coordinates_yco_sorted[0][0]
            y_co_2 = neighbour_coordinates_yco_sorted[1][0]

            pos_xco1 = np.where(legendreAxis_x == x_co_1.item())[0][0]
            pos_xco2 = np.where(legendreAxis_x == x_co_2.item())[0][0]

            pos_yco1 = np.where(legendreAxis_y == y_co_1.item())[0][0]
            pos_yco2 = np.where(legendreAxis_y == y_co_2.item())[0][0]

            Q_ev_11 = imageLegGrad_GradBilinearInterpol[pos_xco1][pos_yco1]
            Q_ev_12 = imageLegGrad_GradBilinearInterpol[pos_xco1][pos_yco2]
            Q_ev_21 = imageLegGrad_GradBilinearInterpol[pos_xco2][pos_yco1]
            Q_ev_22 = imageLegGrad_GradBilinearInterpol[pos_xco2][pos_yco2]

            term_1 = ( ( ( (x_co_2 - x_co) * (y_co_2 - y_co) ) / ( (x_co_2 - x_co_1) * (y_co_2 - y_co_1) ) ) * Q_ev_11)
            term_2 = (( ( (x_co - x_co_1) * (y_co_2 - y_co) ) / ( (x_co_2 - x_co_1) * (y_co_2 - y_co_1) ) ) * Q_ev_21)
            term_3 = ( ( ( (x_co_2 - x_co) * (y_co - y_co_1) ) / ( (x_co_2 - x_co_1) * (y_co_2 - y_co_1) ) ) * Q_ev_12)
            term_4 = ( ( ( (x_co - x_co_1) * (y_co - y_co_1) ) / ( (x_co_2 - x_co_1) * (y_co_2 - y_co_1) ) ) * Q_ev_22)

            req_value = term_1 + term_2 + term_3 + term_4

            Fr1 = torch.tensor([req_value])

            recon_uniform_grid_ = torch.cat((recon_uniform_grid_, Fr1)) 
    sorted_image_ori_size=recon_uniform_grid_.reshape(96,96)

    return sorted_image_ori_size


def get_lejaImage(legendreImage,sortedInds2):
    lejaImage = torch.tensor([])
    for i in sortedInds2:
        for j in sortedInds2:
            sorIm = legendreImage.squeeze(0)[i][j]
            sorIm = torch.tensor([sorIm])
            lejaImage = torch.cat((lejaImage, sorIm)) 
    return lejaImage


def get_sobolev_fitted_recon(AnimageFromBatch,originalAxis_x, originalAxis_y,sortedInds2, X_p, metric_2d, W_param, KsK, u_ob, b, xf):

    lejaImage =get_lejaImage(AnimageFromBatch, sortedInds2) 
    Ksf = matmul(X_p.T, metric_2d(matmul(torch.diag(W_param),lejaImage)))
    w = matmul(KsK.inverse(), Ksf)
    u_ob.set_weights(w)
    X_test = u_ob.data_axes([b,xf]).T
    #X_final = u_ob.data_axes([x,x]).T
    pred = u_ob(X_test).T[0].reshape(len(b),len(xf)).detach().numpy()
    sobolevImage = pred.reshape(len(originalAxis_x),len(originalAxis_y))

    return sobolevImage


#reconstructing using gradient sampling method. Applicable to only images embedded using gradient sampling method
def gradSampledReconstruction( original_image, gradSampledImage, embeddedGridLength_x, embeddedGridLength_y, originalAxis_x, originalAxis_y):

    imageOnSmartGrid,ascen_ground_inds,ascen_wall_inds = get_image_on_smartGrid(original_image, embeddedGridLength_x, embeddedGridLength_y, originalAxis_x, originalAxis_y)
    #print('imageOnSmartGrid.shape',imageOnSmartGrid.shape)
    f1 = interpolate.interp2d(ascen_ground_inds[0], ascen_wall_inds[0], gradSampledImage, kind = 'cubic')
    ImageFromInterpolatedFuncInRegularGrid = f1(originalAxis_x, originalAxis_y)

    return ImageFromInterpolatedFuncInRegularGrid



#def decode_the_batch(batch_ori , batch_encd,legendreAxis_x, legendreAxis_y, originalAxis_x, originalAxis_y, method):

def decode_the_batch(batch_ori , batch_encd, embeddedGridLength_x, embeddedGridLength_y, method):
    originalAxis_x = np.linspace(0, batch_ori.shape[2]-1, batch_ori.shape[2])
    originalAxis_y = np.linspace(0, batch_ori.shape[3]-1, batch_ori.shape[3])

    legendreGridX_length = embeddedGridLength_x
    legendreGridY_length = embeddedGridLength_y

    deg_x = legendreGridX_length
    deg_y = legendreGridY_length
    q_x, _ = np.polynomial.legendre.leggauss(deg_x)
    q_y, _ = np.polynomial.legendre.leggauss(deg_y)
    legendreAxis_x = batch_ori.shape[2]/2 *(q_x+1)
    legendreAxis_y = batch_ori.shape[2]/2 *(q_y+1)


    Q_exact_sorted_x = legendreAxis_x.reshape(-1,1)
    knn_x = NearestNeighbors(n_neighbors=2)
    knn_x.fit(Q_exact_sorted_x)

    Q_exact_sorted_y = legendreAxis_y.reshape(-1,1)
    knn_y = NearestNeighbors(n_neighbors=2)
    knn_y.fit(Q_exact_sorted_y)

    if(method=='sobolev'):
        deg_quad = embeddedGridLength_x-1
        rect = rect = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        diffeo_param = hyper_rect(*rect)
        sob_param = Sobolev(deg=deg_quad, dim=2)
        sob_param.set_s(0)
        x_plt, _, _, x, _, _ = sob_param.get_quad()
        metric_param = sob_param.metric()
        W_param = sob_param.get_leja_weights()
        u_ob = jmp_solver.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
        metric_2d = sob_param.metric(weak=True)
        x_l = sob_param.get_xs()
        X_p = u_ob.data_axes([x,x]).T

        Q_exact = 96/2 *(x+1)
        _, sortedInds = torch.sort(torch.tensor(Q_exact))
        _, sortedInds2 = torch.sort(torch.tensor(sortedInds))       

        K = torch.eye(len(X_p))
        KsK = matmul(X_p.T, metric_2d(matmul(torch.diag(W_param),X_p)))

        b = np.linspace(-1,1,len(originalAxis_x))
        xf= np.linspace(-1,1,len(originalAxis_y))         

    
    batch_Dec = torch.tensor([])
    for i in range(batch_ori.shape[0]):
        if(method == 'local_bilinear'):
            imagereconstructed = imageReconInUniformGrid_bilinearMethod(legendreAxis_x, legendreAxis_y, originalAxis_x, originalAxis_y, batch_encd[i].squeeze(0).cpu(), knn_x, knn_y, Q_exact_sorted_x, Q_exact_sorted_y)
            imagereconstructed = imagereconstructed.clone().detach().unsqueeze(0)
        elif(method=='scipy'):
            imagereconstructed = getDecodingUsingScipy(batch_encd[i].squeeze(0).cpu(), legendreAxis_x, legendreAxis_y, originalAxis_x, originalAxis_y)
            imagereconstructed = torch.tensor(imagereconstructed).unsqueeze(0) 
        elif(method=='sobolev'):
            imagereconstructed = get_sobolev_fitted_recon(batch_encd[i],originalAxis_x, originalAxis_y, sortedInds2, X_p, metric_2d, W_param, KsK, u_ob, b, xf)
            imagereconstructed = torch.tensor(imagereconstructed).unsqueeze(0)
            #print("Sobolev method under progress...")
        elif(method=='grad_sampled'):
            imagereconstructed = gradSampledReconstruction(batch_ori[i].squeeze(0).cpu(), batch_encd[i].squeeze(0).cpu(), embeddedGridLength_x, embeddedGridLength_y, originalAxis_x, originalAxis_y)
            imagereconstructed = torch.tensor(imagereconstructed).unsqueeze(0)
        else:
            print("Check the spelling of the string for method or insert a method")
        #imagereconstructed = imagereconstructed.clone().detach().unsqueeze(0) 
        batch_Dec = torch.cat((batch_Dec,imagereconstructed.unsqueeze(0)))
    return batch_Dec