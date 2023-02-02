import sympy as sp
#import minterpy as mp
import numpy as np
#from minterpy.pointcloud_utils import *

import torch

import random
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from operator import itemgetter
from sklearn.neighbors import NearestNeighbors

from vedo import *

walk = True
no_walk = False

def give_centeroid(arr, grid_dim):
    return np.mean(arr,0).reshape(1, grid_dim)

def give_next_neighbours_barycenter_indices(batch_x, input_barycenter, remaining_indices, sweep_radsius, grid_dim):
    
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
  #print('where_is_it[0]', where_is_it[0])
  #print('where_is_it[0][-1]', where_is_it[0][-1])
  #print('num_neighbours',num_neighbours)
  #break
  #print('distance_cum_index',distance_cum_index[:,1])
  remaining_indices = distance_cum_index[:,1]

  remaining_indices = remaining_indices.astype(int)
  
    
  A = np.array([])
  for i in range(num_neighbours):
    if(i >= distance_cum_index[:,1].shape[0]):
        break
    A = np.concatenate((A, batch_x[int(distance_cum_index[:,1][i])]), axis = 0 )
  
  A = A.reshape(int(A.shape[0]/grid_dim) , grid_dim)
  
  #print("The shape of A is ")
  #print(A.shape)
  if(walk):
    next_barycenter = give_centeroid(A, grid_dim) 
    #print('next_barycenter.shape', next_barycenter.shape)
  else:
    next_barycenter = A[0].reshape(1,2)
    #print('A[0].shape', A[0].shape)
    
  next_barycenter = np.array(next_barycenter)
  #print(next_barycenter.shape)
  next_barycenter = next_barycenter.reshape(next_barycenter.shape[0]* next_barycenter.shape[1])
  #print(next_barycenter)  
  return A, next_barycenter, remaining_indices,num_neighbours


def get_convergent_barycenters(point_cloud, initial_pt,sweep_radsius):   
    
    
    grid_dim = point_cloud.shape[-1]
    print(grid_dim)
    #no_neighbours = int(point_cloud.shape[0] / no_of_barycenetrs_required)
    #initiating no of neighbours
    #no_neighbours = 5
    
    #num_neighbours = int(batch_x.shape[0] / no_of_barycenetrs_required)

    
    bary = initial_pt
    rem_indices = np.array(range(0,point_cloud.shape[0]))
    #print(rem_indices)
    #print("Size of batch : ", point_cloud.shape[0])
    sampled_barycenters = np.array([])
    sampled_barycenters = torch.tensor(sampled_barycenters)
    sampled_barycenters_indices = []

    covered_indices = np.array([])

    for i in range(int(point_cloud.shape[0])):

        if(len(rem_indices) == 2):
            #print("END")
            break

        #print("Iteration number : ", i+1)
        #print("Input barycenter : ")


        old_bary = bary


        #print('rem_indices before', rem_indices)
        neighbours, bary, rem_indices, no_neighbours = give_next_neighbours_barycenter_indices(point_cloud, bary, rem_indices, sweep_radsius, grid_dim)
        #print('no_neighbours', no_neighbours)
        #break
        #print('old_bary.shape', old_bary.shape)
        #print('bary.shape', bary.shape)
        wassDistance = np.linalg.norm(old_bary - bary)  

        covered_indices = np.concatenate((covered_indices, rem_indices[:4] ) ,axis = 0)


        #print("Tracking distance between new barycenter and previous barycenter : ",wassDistance )
        if(wassDistance < 0.0000000001):
            bary = neighbours[0]
            #nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(neighbours)
            #offset_distances, indices = nbrs.kneighbors(bary.reshape(1,2))
            
            #print('indices', indices)
            
            unique_covered_indices = np.unique(covered_indices, axis=0)
            sampled_barycenters = torch.cat((sampled_barycenters, torch.tensor(bary)), 0)
            sampled_barycenters_indices.append(rem_indices[0])

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
    sampled_barycenters = sampled_barycenters.reshape(int((sampled_barycenters.shape[0]/grid_dim)),grid_dim)
    
    return sampled_barycenters, sampled_barycenters_indices

def _compute_distance_matrix(x, p=2):
    x_flat = x.view(x.size(0), -1)

    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)

    return distances

