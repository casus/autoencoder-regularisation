#All Imports
from get_data import get_data, get_data_train, get_data_val
import torch
import os
import numpy as np

import matplotlib.pyplot as plt

from datasets import InMemDataLoader
import torch.nn.functional as F


import sys
import os, os.path
import pandas as pd
import numpy as np
import json
# minterpy
import minterpy as mp

import plotly.express as px
import plotly.graph_objects as go

from matplotlib import pyplot as plt

import pkg as util

import numpy as np
import minterpy as mp
from minterpy.extras.regression import *
from matplotlib import pyplot as plt
import nibabel as nib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("all imports done")

d ='/bigdata/hplsim/aipp/RLtract/deepFibreTracking/examples/data/HCP_extended/'
all_paths = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]
all_paths = [p + '/T1w/T1w_acpc_dc_restore_1.25.nii.gz' for p in all_paths]

print('paths established')


def get_data(paths, device, shuffle=False):
    data = _normalize([_rotate(nib.load(p).get_fdata()) for p in paths])   # load and preprocess all slices from all patients                               
    data_t = torch.FloatTensor(data).to(device)                            # data_t has now the shape: (num_patients, x, y, num_slices)
    data_t = data_t.permute(0, 3, 1, 2)                                    # permute data_t to be in shape (num_patients, num_slices, x, y)
    data_t = data_t[:, :123, :, :]
    data_t = data_t.reshape(data_t.shape[0]*data_t.shape[1],               # reduce dim of data_t to have shape (num_patients*num_slices, x, y)
             data_t.shape[2], data_t.shape[3])
    data_t = data_t.unsqueeze(1)                                           # add image channel, data_t now has shape (num_patients*num_slices, num_channel, x, y)
    if shuffle:                                                            # randomly shuffle all slices
        random_indices = torch.randperm(data_t.shape[0])                   # get list of random indices 
        data_t = data_t[random_indices, :, :, :]                           # reorder the set with the random indices
    return data_t

def _normalize(data):
    data = (data - np.min(data))/(np.max(data)-np.min(data))                # normalize data
    return data                     

def _rotate(data):
    data = np.rot90(data)                                                   # rotate by 90°, without rotation the base of the skull is located to the left of the image       
    return data                                                             # with rotation, the base of the skull is located to the bottom of the image


def get_train_test_set(paths, device, batch_size=32, train_set_size=0.2, test_set_size=0.2):
    assert train_set_size + test_set_size <= 1., "Train and test set size should not exceed 100%"
    
    path_indices = np.arange(len(paths))
    #np.random.shuffle(path_indices)                             # randomize indices of the paths for train and test set selection
    
    num_train = int(np.round_(len(paths) * train_set_size))     # calc amount of training sets to load
    num_test = int(np.round_(len(paths) * test_set_size))       # calc amount of test sets to load
    train_indices = path_indices[:num_train]                    # select unique and random indices from all paths
    test_indices = path_indices[-num_test:]   # for train and test set


    train_data = get_data([paths[i] for i in train_indices], device)  # only load specific indices preveiously selected
    #print('train_data.shape',train_data.shape)
    #torch.save(train_data, '/home/ramana44/regularizedautoencoder-anwi_mrt_another_wass_epc3/lag_coeffs_saved/train_data.pt')

    test_data = get_data([paths[i] for i in test_indices], device)
    print('test_data.shape',test_data.shape)

    train_loader = InMemDataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) # init dataloader for train and test set
    test_loader = InMemDataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True) 
    
    return train_data, test_data
    #return train_loader, test_loader


#input_batch = batch_x

def get_image_lagrange_coeffs(input_image):

    function_vals = input_image
    function_vals_flat = function_vals.reshape(function_vals.shape[0]*function_vals.shape[1])

    sdt = ScatteredDecompositionTree(geo, function_vals_flat,
                                    poly_degree=8, lp_degree=2.0)
    sdt.subdivide([18,18])
    reg_tree = mp.windowed_regression_scattered(sdt)
    lag_poly = merge_tree(reg_tree, sdt, poly_degree = 30)
    lag_coeffs_t = lag_poly.coeffs
    return lag_coeffs_t


print('all functions defined')

#train_loader, test_loader = get_train_test_set(all_paths, device, batch_size=7626, train_set_size=0.8)

_, batch_x = get_train_test_set(all_paths, device, batch_size=7626, train_set_size=0.8)

print('train and test data alloted ')
#batch_x = next(iter(test_loader))

print('batch_x.shape',batch_x.shape)


#precalculations before function call

lat = np.array([[i]*174 for i in range(145)]) 
lon = np.array([list(range(145)) for i in range(174)])
height, width = batch_x[0][0].shape
grid_len = height * width
geo = np.hstack([lon.reshape((grid_len, 1)), lat.reshape((grid_len, 1))])

print("precalculations done")
test_lag_coeffs = torch.tensor([])

for i in range(batch_x.shape[0]):
    check = get_image_lagrange_coeffs(batch_x[i][0])
    check_t = torch.tensor(check)
    test_lag_coeffs = torch.cat((test_lag_coeffs,check_t), 0)
    #check = 0 
    #print('check_t.shape',check_t.shape)
    if(i%500==0):
        print('batch_lag_coeffs', test_lag_coeffs.shape)

print()
sp1=int(test_lag_coeffs.shape[0]/check_t.shape[0])
sp2 = 1
sp3 = check_t.shape[0]
print('sp1, sp2, sp3', sp1, sp2, sp3)

test_lag_coeffs_indiv = test_lag_coeffs.reshape(sp1,sp2,sp3)
print('batch_lag_coeffs_indiv.shape', test_lag_coeffs_indiv.shape)

print()
print('now saving lagrange coeffs..')
torch.save(test_lag_coeffs_indiv, '/home/ramana44/regularizedautoencoder-anwi_mrt_another_wass_epc3/lag_coeffs_saved/test_lag_coeffs_indiv_deg30.pt')

print()
print('loading saved coeffs..')
loaded_coeffs_tst = torch.load('/home/ramana44/regularizedautoencoder-anwi_mrt_another_wass_epc3/lag_coeffs_saved/test_lag_coeffs_indiv_deg30.pt')
print()
print('printing shape of loaded coeffs', loaded_coeffs_tst.shape)