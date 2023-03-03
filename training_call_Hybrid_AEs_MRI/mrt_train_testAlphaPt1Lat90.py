import sys
sys.path.append('/home/ramana44/autoencoder-regularisation-')

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
    assert train_set_size + test_set_size <= 1., "Train and test set size should not exceed 100%"
    
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


### run training and monitor on W&B
from train_ae_n import train
from activations import Sin

import wandb
# wandb.login()

os.environ['WANDB_API_KEY'] = 'e1a3b85ef17f1f7e3a683f4dd0d1fcbacb4668b1'
wandb.login()

wandb.init(project='Test_mrt', entity='ae_reg_team')




#config = wandb.config
reg_nodes_sampling = "legendre"
alpha = 0.1
hidden_size = 1000
deg_poly = 20
latent_dim = 90
lr = 1e-4
no_layers = 3
train_set_size = 0.05

#chethan 
config_defaults = {
            'reg_nodes_sampling': reg_nodes_sampling,
            'alpha': alpha,
            'hidden_size': hidden_size,
            'deg_poly': deg_poly,
            'latent_dim': latent_dim,
            'lr': lr,
            'no_layers' : no_layers,
            'train_set_size' : train_set_size
        }
wandb.init(project='Test_mrt', entity='ae_reg_team', config=config_defaults)

train_loader, test_loader = get_train_test_set(all_paths, device, train_set_size=train_set_size, batch_size=200)

model, model_reg, loss_arr_reg, loss_arr_reco, loss_arr_base, loss_arr_val_reco, loss_arr_val_base = train(train_loader, test_loader, no_epochs=50, reco_loss="mse", latent_dim=latent_dim, 
          hidden_size=hidden_size, no_layers=no_layers, activation = Sin(), lr=lr, alpha = alpha, bl=False,
          seed = 2342, train_base_model=True, no_samples=20, deg_poly=deg_poly,
          reg_nodes_sampling=reg_nodes_sampling, no_val_samples = 10, use_guidance = False, train_set_size=train_set_size,
          enable_wandb=False, wandb_project='Test_mrt', wandb_entity='ae_reg_team')







