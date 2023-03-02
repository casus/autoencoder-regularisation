from get_data import get_data
import torch
import os
import numpy as np

from datasets import InMemDataLoader


#### create list of all nifti file paths ###
#d ='/bigdata/hplsim/aipp/RLtract/deepFibreTracking/examples/data/HCP_extended/'
'''d = '/bigdata/hplsim/aipp/RLtract/deepFibreTracking/examples/data/HCP_extended/'
all_paths = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]
all_paths = [p + '/T1w/T1w_acpc_dc_restore_1.25.nii.gz' for p in all_paths]'''

'''d = '/home/ramana44/all_scans_single_channel_equal_dim/'
all_paths = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]
all_paths = [p for p in all_paths]'''
all_paths = []
for root, dirs, files in os.walk(os.path.abspath("/home/ramana44/all_scans_single_channel_equal_dim/")):
    for file in files:
        #print(os.path.join(root, file))
        all_paths.append((os.path.join(root, file)))

print('len(all_paths)',len(all_paths))


### load data ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_train_test_set(paths, device, batch_size=200, train_set_size=0.8, test_set_size=0.2):
    #assert train_set_size + test_set_size <= 1., "Train and test set size should not exceed 100%"
    
    path_indices = np.arange(len(paths))
    #np.random.shuffle(path_indices)                             # randomize indices of the paths for train and test set selection
    
    num_train = int(np.round_(len(paths) * train_set_size))     # calc amount of training sets to load
    num_test = int(np.round_(len(paths) * test_set_size))       # calc amount of test sets to load
    train_indices = path_indices[:num_train]                    # select unique and random indices from all paths
    test_indices = path_indices[-num_test:]                     # for train and test set


    train_data = get_data([paths[i] for i in train_indices], device)  # only load specific indices preveiously selected
    test_data = get_data([paths[i] for i in test_indices], device)
    print('train_data.shape',train_data.shape)

    train_loader = InMemDataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) # init dataloader for train and test set
    test_loader = InMemDataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True) 
    return train_loader, test_loader



reg_nodes_sampling = "legendre"
alpha = 0.5
hidden_size = 1000
deg_poly = 20
latent_dim = 80
lr = 1e-4
no_layers = 5
train_set_size = 1

train_loader, test_loader = get_train_test_set(all_paths, device, train_set_size=train_set_size, batch_size=61275)

#batch_x = next(iter(train_loader))

#print('batch_x.shape',batch_x.shape)


#batch_x_t = next(iter(test_loader))

#print('batch_x.shape',batch_x_t.shape)


for inum, batch_x in enumerate(train_loader):
    #global_step += 1
    loss_C1 = torch.FloatTensor([0.]).to('cuda') 
    # plain reconstruction using AE
    #batch_x = get_encoded_batch(batch_x,Q_exact)
    #batch_x = get_SmartGridBatch(batch_x,smart_indsX, smart_indsY)
    #batch_x = batch_x.float()
    batch_x = batch_x.to('cuda')
    #print('training batch size: ', batch_x.shape)

    #batch_x[torch.where(batch_x < 0)] = 0
    #batch_x = batch_x / batch_x.max()

    #reconstruction = model_reg(batch_x)
    #reconstruction = reconstruction.view(batch_x.shape)
    print(batch_x.shape)
    torch.save(batch_x[:50000], '/home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/trainDataSet.pt')
    torch.save(batch_x[50000:], '/home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/testDataSet.pt')


trdst = torch.load('/home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/trainDataSet.pt')
tsdst = torch.load('/home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/testDataSet.pt')
print('trdst.shape',trdst.shape)
print('tsdst.shape',tsdst.shape)