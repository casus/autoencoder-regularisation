from get_data import get_data
import torch
import os
import numpy as np

from datasets import InMemDataLoader



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


noTrainImagesToSave = 10
noTestImagesToSave = 2

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
    print("printing")
    print(batch_x.shape)
    torch.save(batch_x[:noTrainImagesToSave], '/home/ramana44/autoencoder-regularisation-/savedData/trainDataSet.pt')
    torch.save(batch_x[noTrainImagesToSave:noTrainImagesToSave+noTestImagesToSave], '/home/ramana44/autoencoder-regularisation-/savedData/testDataSet.pt')


trdst = torch.load('/home/ramana44/autoencoder-regularisation-/savedData/trainDataSet.pt')
tsdst = torch.load('/home/ramana44/autoencoder-regularisation-/savedData/testDataSet.pt')


print('trdst.shape',trdst.shape)
print('tsdst.shape',tsdst.shape)

