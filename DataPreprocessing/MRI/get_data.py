#!/usr/bin/python
# -*- coding: latin-1 -*-

import nibabel as nib
import numpy as np
import torch

def get_data(paths, device, shuffle=False):
    #print('before normalize', [_rotate(nib.load(p).get_fdata()) for p in paths].shape)
    data = _normalize([_rotate(nib.load(p).get_fdata()) for p in paths])   # load and preprocess all slices from all patients                               
    data_t = torch.FloatTensor(data).to(device)                            # data_t has now the shape: (num_patients, x, y, num_slices)
    print('initial',data_t.shape)
    data_t = data_t.permute(0, 3, 1, 2)                                    # permute data_t to be in shape (num_patients, num_slices, x, y)
    print('after permute', data_t.shape)
    data_t = data_t[:, 20:45, :, :]                                         # clean outliers at the end of a scan 
    print('after 123', data_t.shape)
    data_t = data_t.reshape(data_t.shape[0]*data_t.shape[1],               # reduce dim of data_t to have shape (num_patients*num_slices, x, y)
             data_t.shape[2], data_t.shape[3])
    print('after reshape', data_t.shape)
    data_t = data_t.unsqueeze(1)                                           # add image channel, data_t now has shape (num_patients*num_slices, num_channel, x, y)
    print('after unsqueeze',data_t.shape)
    if shuffle:                                                            # randomly shuffle all slices
        random_indices = torch.randperm(data_t.shape[0])                   # get list of random indices 
        data_t = data_t[random_indices, :, :, :]                           # reorder the set with the random indices
        print('after shuffle', data_t.shape)
    return data_t

def _normalize(data):
    print('before normalize',len(data) )
    data = (data - np.min(data))/(np.max(data)-np.min(data))                # normalize data
    print('after normalize',len(data) )
    return data                     

def _rotate(data):
    #print('before rotate', data.shape)
    data = np.rot90(data)                                                   # rotate by 90°, without rotation the base of the skull is located to the left of the image       
    #print('after rotate', data.shape)
    return data                                                             # with rotation, the base of the skull is located to the bottom of the image

def get_data_train(path_to_data, path_to_indx):
    data = nib.load(path_to_data).get_fdata()
    data_norm = (data - np.min(data))/(np.max(data)-np.min(data))
    indx = np.load(path_to_indx)
    train_indx = np.delete(np.array(list(range(0, 145))), indx)
    data_t = torch.from_numpy(data_norm[:, :, train_indx]).float().to('cuda')
    inp = torch.transpose(data_t, 1, 2)
    inp = torch.transpose(inp, 1, 0)
    return inp

def get_data_val(path_to_data, path_to_indx):
    data = nib.load(path_to_data).get_fdata()
    data_norm = (data - np.min(data))/(np.max(data)-np.min(data))
    indx = np.load(path_to_indx)
    data_t = torch.from_numpy(data_norm[:, :, indx]).float().to('cuda')
    inp = torch.transpose(data_t, 1, 2)
    inp = torch.transpose(inp, 1, 0)
    return inp

    