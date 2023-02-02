import sys
sys.path.append('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders')
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
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
from scipy import interpolate
import ot
import jmp_solver1.surrogates
import matplotlib
matplotlib.rcdefaults() 
from matplotlib.pyplot import figure
from operator import itemgetter

#from ripser import ripser
import ripser
import persim
from persim import plot_diagrams

Analys_size = 2000


from models_un import AE_un
from models import AE
from activations import Sin

def give_centeroid(arr):
    
    #length = arr.shape[0]
    numCoords = arr.shape[1]
    #print('arr.shape', arr.shape)
    cetroid = np.mean(arr, 0).reshape(1, numCoords)
    #print('cetroid.shape',cetroid.shape)
    
    #cetroid1 = np.sum(arr, 0).reshape(1, numCoords)/length
    #print('cetroid1.shape',cetroid1.shape)
    
    
    #print('cetroid', cetroid)
    
    #print('cetroid1', cetroid1)
    
    #sum_x = np.sum(arr[:, 0])
    #sum_y = np.sum(arr[:, 1])
    #sum_z = np.sum(arr[:, 2])
    #return np.array([[sum_x/length, sum_y/length, sum_z/length]])

    return cetroid

def give_next_neighbours_barycenter_indices(batch_x, input_barycenter, remaining_indices, sweep_radsius):
    
  #sweep_radsius = 0.2
  #num_neighbours = int(batch_x.shape[0] / no_of_barycenetrs_required)
    
  wasserDistance = []
  distance_cum_index = np.array([])

  for j in remaining_indices:
    
    wassDistance = dist = np.linalg.norm(batch_x[j]-input_barycenter)

    
    distance_cum_index = np.concatenate((distance_cum_index, np.array([wassDistance, j])), axis = 0)
   
  distance_cum_index = distance_cum_index.reshape(int(distance_cum_index.shape[0]/2), 2)
  distance_cum_index = sorted(distance_cum_index, key=itemgetter(0))
  
  distance_cum_index = np.array(distance_cum_index)
  only_distances = distance_cum_index[:,0]
  #print(only_distances)
  where_is_it = np.where( only_distances < sweep_radsius ) 
  #print('where_is_it', where_is_it)  
  #print('where_is_it[0][-1]', where_is_it[0][-1])
  num_neighbours = where_is_it[0][-1] +1
  
  remaining_indices = distance_cum_index[:,1]

  remaining_indices = remaining_indices.astype(int)
  
    
  A = np.array([])
  for i in range(num_neighbours):
    if(i >= distance_cum_index[:,1].shape[0]):
        break
    A = np.concatenate((A, batch_x[int(distance_cum_index[:,1][i])]), axis = 0 )
  
  mul_dim = batch_x.shape[-1]
  A = A.reshape(int(A.shape[0]/mul_dim) , mul_dim)
  
  #print("The shape of A is ")
  #print(A.shape)
  next_barycenter = give_centeroid(A) 
  #print('next_barycenter.shape', next_barycenter.shape)
  next_barycenter = np.array(next_barycenter)
  #print(next_barycenter.shape)
  next_barycenter = next_barycenter.reshape(next_barycenter.shape[0]* next_barycenter.shape[1])
  #print(next_barycenter)  
  return A, next_barycenter, remaining_indices,num_neighbours


def get_convergent_barycenters(point_cloud, initial_pt,sweep_radsius):   
    
    #no_neighbours = int(point_cloud.shape[0] / no_of_barycenetrs_required)
    #initiating no of neighbours
    #no_neighbours = 5
    
    #num_neighbours = int(batch_x.shape[0] / no_of_barycenetrs_required)

    
    bary = initial_pt
    rem_indices = np.array(range(0,point_cloud.shape[0]))
    #print("Size of batch : ", point_cloud.shape[0])
    sampled_barycenters = np.array([])
    sampled_barycenters = torch.tensor(sampled_barycenters)
    covered_indices = np.array([])

    for i in range(int(point_cloud.shape[0])):

        if(len(rem_indices) == 2):
            #print("END")
            break

        #print("Iteration number : ", i+1)
        #print("Input barycenter : ")


        old_bary = bary


        #print('rem_indices before', rem_indices)
        neighbours, bary, rem_indices, no_neighbours = give_next_neighbours_barycenter_indices(point_cloud, bary, rem_indices, sweep_radsius)
        #print('no_neighbours', no_neighbours)

        wassDistance = np.linalg.norm(old_bary - bary)  

        covered_indices = np.concatenate((covered_indices, rem_indices[:4] ) ,axis = 0)


        #print("Tracking distance between new barycenter and previous barycenter : ",wassDistance )
        if(wassDistance < 0.000001):

            unique_covered_indices = np.unique(covered_indices, axis=0)
            sampled_barycenters = torch.cat((sampled_barycenters, torch.tensor(bary)), 0)

            s1 = set(rem_indices)
            s2 = set(unique_covered_indices)
            rem_set = s1 - s2
            rem_inds = list(rem_set)
            #rem_indices = rem_set
            rem_indices = rem_indices[no_neighbours:]

            #print("Sampled barycenters are")
            #print(sampled_barycenters)
            #no_neighbours = 5
            #print('rem_indices',rem_indices)
            #print("len(rem_indices)",len(rem_indices))
            if(len(rem_indices) ==0):
                break
            bary = point_cloud[rem_indices[0]]
    mul_dim = point_cloud.shape[-1]
    sampled_barycenters = sampled_barycenters.reshape(int((sampled_barycenters.shape[0]/mul_dim)),mul_dim)
    
    return sampled_barycenters


def _compute_distance_matrix(x, p=2):
    x_flat = x.view(x.size(0), -1)

    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)

    return distances

def get_persistence_diagram(point_cloud, maximum_dim):

    point_cloud = torch.tensor(point_cloud)

    dist_matrix = _compute_distance_matrix(point_cloud, p=2)
    diagrams = ripser.ripser(dist_matrix.cpu().detach().numpy(), distance_matrix=True, maxdim=maximum_dim)['dgms']
    return diagrams, plot_diagrams(diagrams, show=True)






# load trained rAE and bAE
latent_dims = [4]
all_hyb_base_models = []
all_hyb_reg_models = []
all_test_coeffs = []
all_X_p = []
for lat_dim in latent_dims:
    deg_quad = 20
    u_ob = jmp_solver1.surrogates.Polynomial(n=deg_quad,p=np.inf, dim=2)
    x = np.linspace(-1,1,32)
    X_p = u_ob.data_axes([x,x]).T

    testImages = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/testImages.pt',map_location=torch.device('cpu'))
    testCoeffs = torch.load('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_coeffs/coeffs_saved/LSTSQ_testcoeffs_FMNIST_dq'+str(deg_quad)+'.pt',map_location=torch.device('cpu'))

    testImages = testImages[:Analys_size]
    testCoeffs = testCoeffs[:Analys_size]

    path_hyb = '/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders-STORAGE_SPACE/FMNIST_RK_space/output/MRT_full/test_run_saving/'
    path_unhyb = '/home/ramana44/FashionMNIST5LayersTrials/output/MRT_full/test_run_saving/'

    #specify hyperparameters
    reg_nodes_sampling = 'legendre'
    alpha = 0.5
    frac = 0.4
    hidden_size = 100
    deg_poly = 21
    deg_poly_forRK = 21
    latent_dim = lat_dim
    lr = 0.0001
    no_layers = 3
    no_epochs= 100
    name_hyb = '_'+reg_nodes_sampling+'_'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly_forRK)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)
    name_unhyb = '_'+reg_nodes_sampling+'__'+str(frac)+'_'+str(alpha)+'_'+str(hidden_size)+'_'+str(deg_poly)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(no_layers)#+'_'+str(no_epochs)

    inp_dim_hyb = (deg_quad+1)*(deg_quad+1)

    inp_dim_unhyb = [1,32,32]

    model_reg = AE_un(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
    model_base = AE_un(inp_dim_unhyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

    RK_model_reg = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
    RK_model_base = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

    #RK_model_reg.load_state_dict(torch.load(path_hyb+'model_regLSTQS'+str(deg_quad)+''+name_hyb, map_location=torch.device('cpu')))
    #RK_model_base.load_state_dict(torch.load(path_hyb+'model_baseLSTQS'+str(deg_quad)+''+name_hyb, map_location=torch.device('cpu')))

    model_reg.load_state_dict(torch.load(path_unhyb+'model_reg_TDA'+name_unhyb, map_location=torch.device('cpu')))
    model_base.load_state_dict(torch.load(path_unhyb+'model_base_TDA'+name_unhyb, map_location=torch.device('cpu')))

    all_hyb_base_models.append(model_base)
    all_hyb_reg_models.append(model_reg)
    all_test_coeffs.append(testImages)
    all_X_p.append(X_p)



all_rec_rAE_test = []
all_rec_bAE_test = []
for i in range(len(latent_dims)):
    rec_rAE_test = all_hyb_reg_models[i].encoder(all_test_coeffs[i].float())#.view(all_test_coeffs[i].shape)
    rec_bAE_test = all_hyb_base_models[i].encoder(all_test_coeffs[i].float())#.view(all_test_coeffs[i].shape)
    
    rec_rAE_test = torch.tensor(rec_rAE_test, requires_grad=False)
    rec_bAE_test = torch.tensor(rec_bAE_test, requires_grad=False)

    all_rec_rAE_test.append(rec_rAE_test)
    all_rec_bAE_test.append(rec_bAE_test)


print('testImages.shape', testImages.shape)

testImages = testImages.reshape(testImages.shape[0], testImages.shape[1]*testImages.shape[2]*testImages.shape[3])

testImages_barycenters = get_convergent_barycenters(testImages, testImages[0], 6.496533333333332)
#6.4965334
print('testImages_barycenters.shape', testImages_barycenters.shape)


#print(all_rec_rAE_test[0].shape, 'all_rec_rAE_test[0].shape')
print(all_rec_bAE_test[0].shape, 'all_rec_bAE_test[0].shape')

#plt.scatter(all_rec_bAE_test[4][:,0], all_rec_bAE_test[4][:,1])
#plt.close()
def _compute_distance_matrix(x, p=2):
    x_flat = x.view(x.size(0), -1)

    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)

    return distances

#dist_matrix_Lat_10 = _compute_distance_matrix(all_rec_rAE_test[4], p=2)


#from ripser import ripser
import ripser
import persim
from persim import plot_diagrams


#Persistent Homology of first 200 test images using L2 distance matrix
dist_matrix_Lat_ori = _compute_distance_matrix(testImages_barycenters, p=2)
diagrams_ori = ripser.ripser(dist_matrix_Lat_ori.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
plot_diagrams(diagrams_ori, show=True)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('Birth',fontsize=15)
plt.ylabel('Death' ,fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/topological_analysis_in_latent_space/plots_from_topology2/PH_FMNIST_testImages_L2_dist_matrix_L2BarycneterSubsampling_size200.png')
plt.close()

for i in range(len(latent_dims)):
    #print(latent_dims[i])

    #Persistent Homology of first 200 test images in all latent dimensions using L2 distance matrix HybAEREG
    #all_rec_rAE_test_barycenters = get_convergent_barycenters(all_rec_rAE_test[i], all_rec_rAE_test[i][0], latent_dims[i]*0.04)

    all_rec_bAE_test_barycenters = get_convergent_barycenters(all_rec_rAE_test[i], all_rec_rAE_test[i][0], latent_dims[i]*0.05854)

    #print('all_rec_rAE_test_barycenters.shape', all_rec_rAE_test_barycenters.shape)

    print('all_rec_rAE_test_barycenters.shape', all_rec_bAE_test_barycenters.shape)

    #dist_matrix_Lat = _compute_distance_matrix(all_rec_rAE_test_barycenters, p=2)
    dist_matrix_Lat = _compute_distance_matrix(all_rec_bAE_test_barycenters, p=2)
    
    diagrams = ripser.ripser(dist_matrix_Lat.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
    plot_diagrams(diagrams, show=True)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Birth',fontsize=15)
    plt.ylabel('Death' ,fontsize=15)
    plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/topological_analysis_in_latent_space/plots_from_topology2/PH_FMNIST_NonHybAeRegLatDim'+str(latent_dims[i])+'_L2_dist_matrix_L2BarycneterSubsampling_size200.png')
    #plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/topological_analysis_in_latent_space/plots_from_topology2/PH_FMNIST_NonHybMLPAELatDim'+str(latent_dims[i])+'_L2_dist_matrix_L2BarycneterSubsampling_size200.png')
    plt.close()


'''# L2 distances of Persistent homology signatures in latent spaces from the persistent homology signatures of original point cloud of images

all_mse = []
for i in range(len(latent_dims)):
    dist_matrix_Lat = _compute_distance_matrix(all_rec_rAE_test[i], p=2)
    diagrams = ripser.ripser(dist_matrix_Lat.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
    mse = np.mean(np.sqrt((diagrams[0][:-1] - diagrams_ori[0][:-1])**2))
    all_mse.append(mse)
    pre_diag = diagrams
figure(figsize=(8, 6), dpi=100)
plt.plot(latent_dims, all_mse)
plt.xticks(latent_dims, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('Latent Dimension',fontsize=15)
plt.ylabel('L-2 Distance' ,fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/topological_analysis_in_latent_space/plots_from_topology/AllLatDimPH_distances_from_originalImageCloudPH_FMNIST_HybAeReg_L2_dist_matrix_L2BarycneterSubsampling_size200.png')
plt.close()



for i in range(len(latent_dims)):
    #print(latent_dims[i])
    #Persistent Homology of first 200 test images in all latent dimensions using L2 distance matrix HyMLPAE
    dist_matrix_Lat = _compute_distance_matrix(all_rec_bAE_test[i], p=2)
    diagrams = ripser.ripser(dist_matrix_Lat.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
    plot_diagrams(diagrams, show=True)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Birth',fontsize=15)
    plt.ylabel('Death' ,fontsize=15)
    plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/topological_analysis_in_latent_space/plots_from_topology/PH_FMNIST_HybMlpaeLatDim'+str(latent_dims[i])+'_L2_dist_matrix_L2BarycneterSubsampling_size200.png')
    plt.close()


all_mse = []
for i in range(len(latent_dims)):
    dist_matrix_Lat = _compute_distance_matrix(all_rec_bAE_test[i], p=2)
    diagrams = ripser.ripser(dist_matrix_Lat.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
    mse = np.mean(np.sqrt((diagrams[0][:-1] - diagrams_ori[0][:-1])**2))
    all_mse.append(mse)
    pre_diag = diagrams
figure(figsize=(8, 6), dpi=100)
plt.plot(latent_dims, all_mse)
plt.xticks(latent_dims, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('Latent Dimension',fontsize=15)
plt.ylabel('L-2 Distance' ,fontsize=15)
plt.savefig('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/topological_analysis_in_latent_space/plots_from_topology/AllLatDimPH_distances_from_originalImageCloudPH_FMNIST_HybMlpae_L2_dist_matrix_L2BarycneterSubsampling_size200.png')
plt.close()'''