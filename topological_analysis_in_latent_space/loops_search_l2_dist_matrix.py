import sys
sys.path.append('/home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders')


from get_data import get_data, get_data_train, get_data_val
import torch
#import os

#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

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

from models_un import AE_un
from models import AE
from activations import Sin

# All Functions 
def _compute_distance_matrix(x, p=2):
    x_flat = x.view(x.size(0), -1)

    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)

    return distances

# function to check whether the selected edge is going to close a potential loop

def expecting_a_cycle(actual_new_test, my_edge):

    left_ind = my_edge[0][0]
    right_ind = my_edge[0][1]
    found_right_ind = False
    going_nowhere= False

    new_test = actual_new_test

    tracker = 0
    no_branches_formed = True
    while (not(found_right_ind) or not(going_nowhere)):

        positions1 = (new_test == left_ind).nonzero(as_tuple=False)

        if(positions1.shape[0]>1):
            edge_to_delete = new_test[positions1[0][0]]
            no_branches_formed = False
        
        branches_rising = positions1.shape[0]

        if(positions1.shape[0]==0):
            going_nowhere= True
            if(no_branches_formed):
                break
            
            left_ind = my_edge[0][0]

            deletable_edge_position1 = (actual_new_test == edge_to_delete[0]).nonzero(as_tuple=False)
            deletable_edge_position2 = (actual_new_test == edge_to_delete[1]).nonzero(as_tuple=False)

            deletable_edge_position1 = deletable_edge_position1[:,0]

            deletable_edge_position2 = deletable_edge_position2[:,0]

            a_cat_b1, counts1 = torch.cat([deletable_edge_position1, deletable_edge_position2]).unique(return_counts=True)
            deletable_row_position = a_cat_b1[torch.where(counts1.gt(1))]

            if(deletable_row_position.shape[0]==0):
                going_nowhere = True
                break

            deletable_row_position = deletable_row_position[0]

            actual_new_test = torch.cat((actual_new_test[:deletable_row_position], actual_new_test[deletable_row_position+1:]))
            new_test = actual_new_test

            positions1 = (new_test == left_ind).nonzero(as_tuple=False)

            if(tracker ==0):
                break

        if(positions1.shape[0]>1):
            edge_to_delete = new_test[positions1[0][0]]
            no_branches_formed = False
                
        first_position = positions1[0][0]
        adj_edge1 = new_test[positions1[0][0]]
        other_end1 = abs(positions1 - torch.tensor([[0, 1]]))


        consec_pt1 = new_test[other_end1[0][0]][other_end1[0][1]]
        consec_pt1 = int(consec_pt1)

        if(consec_pt1 == right_ind):
            found_right_ind = True
            break

        else:
            left_ind = consec_pt1
            new_test = torch.cat((new_test[:first_position], new_test[first_position+1:]))
            tracker = tracker+1
    
    return found_right_ind


def get_all_edges(dist_matrix):
    
    dist_matrix = torch.unique(dist_matrix, dim=0)
    dist_matrix = torch.unique(dist_matrix, dim=1)

    upp_diag = torch.triu(dist_matrix, diagonal=1)

    ff = upp_diag.sort()

    sorted_upper_diag_edges = ff[0]

    sorted_upper_diag_indices = ff[1]

    flattened_uppdg_edges = torch.flatten(sorted_upper_diag_edges)

    non_zero_flattened_uppdg_edges = flattened_uppdg_edges[flattened_uppdg_edges.nonzero()]

    non_zero_flattened_uppdg_edges = non_zero_flattened_uppdg_edges.reshape(non_zero_flattened_uppdg_edges.shape[0])

    increasing_edges = non_zero_flattened_uppdg_edges.sort()[0]
    increasing_edges = torch.unique(increasing_edges, dim=0)
    
    #print('increasing_edges', increasing_edges)
    
    selected_edges = torch.tensor([])
    dead_indices = torch.tensor([])
    potential_triangles = torch.tensor([])
    edge_leads_to_loop = False

    for i in range(increasing_edges.shape[0]):
        a = (upp_diag == increasing_edges[i]).nonzero(as_tuple=False)

        if(selected_edges.shape[0] > 1):
            edge_leads_to_loop = False #expecting_a_cycle(selected_edges, a)

        if(not(edge_leads_to_loop)):
            selected_edges = torch.cat(((selected_edges, a)), 0)

    #print('selected_edges', selected_edges)
    zeroD_PH = torch.tensor([])
    for i in range(selected_edges.shape[0]):    
        death = dist_matrix[int(selected_edges[i][0])][int(selected_edges[i][1])]
        death = death.reshape(1,1)    
        zeroD_PH = torch.cat(((zeroD_PH, death)), 0)

    births = torch.zeros(zeroD_PH.shape[0], 1)
    zeroD_PH_births_deaths = torch.cat((births, zeroD_PH ),1)

    return selected_edges, zeroD_PH_births_deaths

    
def indices_array_any_size(m, n):
    r = np.arange(n)
    s = np.arange(m)
    out = np.empty((n,m,2),dtype=int)
    out[:,:,0] = r[:,None]
    out[:,:,1] = s
    output = out.reshape(m*n,2)
    output = torch.tensor(output).type(torch.FloatTensor)

    return output


#Computationally efficient than  previous method to calculate M 
def compute_M(img_indices, p=2):
    x = img_indices
    x_flat = x.view(x.size(0), -1)
    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
    
    return distances**2


def wass_distance(image1,image2, M_matrix, reg):
    gs = ((image1 + 10**(-6)).reshape(M_matrix.shape[0],1)) / torch.sum((image1))
    h = ((image2 + 10**(-6)).reshape(M_matrix.shape[0],1)) / torch.sum((image2))
    # 10**(-10) added to avoid numerical errors in sinkhorn
    wassDistance = ot.sinkhorn2(h, gs, M_matrix, reg)
    #0.04 is the regularization parameter. You can play around with it 
    return wassDistance


# function to get unbranched edges from the other side till there is branch

def right_side_pts_before_branching(actual_new_test, my_edge):

    left_ind = my_edge[0][1]
    right_ind = my_edge[0][0]
    found_right_ind = False
    going_nowhere= False

    new_test = actual_new_test
    actual_new_test_an = actual_new_test
    
    tracker = 0
    no_branches_formed = True
    loop_tracker = 0
    positions1 = (new_test == left_ind).nonzero(as_tuple=False)
    loops_collec = []
    current_loop = torch.tensor([])
    consec_pt_tracker = torch.tensor([])
    while (not(found_right_ind) or not(going_nowhere)):

        positions1 = (new_test == left_ind).nonzero(as_tuple=False)
        
        if(positions1.shape[0]>1):
            break

        branches_rising = positions1.shape[0]

        if(positions1.shape[0]==0):
            #lets see
            break

        else:
            first_position = positions1[0][0]

            adj_edge1 = new_test[positions1[0][0]]
            other_end1 = abs(positions1 - torch.tensor([[0, 1]]))


            consec_pt1 = new_test[other_end1[0][0]][other_end1[0][1]]
            consec_pt1s = torch.unsqueeze(consec_pt1,0)
            consec_pt_tracker = torch.cat((consec_pt_tracker, consec_pt1s),0)
            consec_pt1 = int(consec_pt1)

            current_loop = torch.cat((current_loop,adj_edge1),0)
            current_loop1 = current_loop.reshape(int(current_loop.shape[0]/2),2)
            
            if(consec_pt1 == my_edge[0][0]):
                current_loop = torch.tensor([])
                
            if(consec_pt1 == right_ind):
                my_edge1 = torch.squeeze(my_edge,0)
                current_loop = torch.cat((current_loop,my_edge1),0)
                current_loop1 = current_loop.reshape(int(current_loop.shape[0]/2),2)                

                loops_collec.append(current_loop1)

                loop_tracker = loop_tracker + 1
            left_ind = consec_pt1
            new_test = torch.cat((new_test[:first_position], new_test[first_position+1:]))
            tracker = tracker+1
    
    return consec_pt_tracker

# function to check whether the selected edge is going to close a potential loop

def get_all_loops_formed(actual_new_test, my_edge):
    edge_to_delete = torch.tensor([ np.inf, np.inf])
    other_side_unbranched_pts = right_side_pts_before_branching(actual_new_test, my_edge)
    #print('other_side_unbranched_pts.shape',other_side_unbranched_pts.shape[0])
    left_ind = my_edge[0][0]
    right_ind = my_edge[0][1]
    found_right_ind = False
    going_nowhere= False

    new_test = actual_new_test
    actual_new_test_an = actual_new_test
    
    tracker = 0
    no_branches_formed = True
    loop_tracker = 0
    positions1 = (new_test == left_ind).nonzero(as_tuple=False)
    loops_collec = []
    current_loop = torch.tensor([])
    consec_pt_tracker = torch.tensor([])
    while (not(found_right_ind) or not(going_nowhere)):

        positions1 = (new_test == left_ind).nonzero(as_tuple=False)
        #print(new_test)
        #print()
        #print('positions1.shape[0]',positions1.shape[0])
        #print()
        
        if(positions1.shape[0]>1):
            #edg_q_del = new_test[positions1[0][0]]
            other_end_con = abs(positions1 - torch.tensor([[0, 1]]))
            consec_pt_con = new_test[other_end_con[0][0]][other_end_con[0][1]]
            #print('did i get consec_pt_con ', consec_pt_con)
            #print('now check if it works', not(consec_pt_con in consec_pt_tracker))
            
            if(not(other_side_unbranched_pts.shape[0] == 0)):
                if(not(consec_pt_con in consec_pt_tracker) and not(consec_pt_con==other_side_unbranched_pts[0])):
                    edge_to_delete = new_test[positions1[0][0]]
            else:
                if(not(consec_pt_con in consec_pt_tracker)):
                    edge_to_delete = new_test[positions1[0][0]]                
            no_branches_formed = False
            #print('edge_to_delete first',edge_to_delete)
        branches_rising = positions1.shape[0]

        if(positions1.shape[0]==0):
            current_loop = torch.tensor([])
            consec_pt_tracker = torch.tensor([])
            #going_nowhere= True
            '''if(no_branches_formed):
                break'''
            
            left_ind = my_edge[0][0]

            deletable_edge_position1 = (actual_new_test == edge_to_delete[0]).nonzero(as_tuple=False)
            deletable_edge_position2 = (actual_new_test == edge_to_delete[1]).nonzero(as_tuple=False)

            deletable_edge_position1 = deletable_edge_position1[:,0]

            deletable_edge_position2 = deletable_edge_position2[:,0]

            a_cat_b1, counts1 = torch.cat([deletable_edge_position1, deletable_edge_position2]).unique(return_counts=True)
            deletable_row_position = a_cat_b1[torch.where(counts1.gt(1))]
            #print()
            #print('deletable_row_position',deletable_row_position)
            
            if(deletable_row_position.shape[0]==0):
                #going_nowhere = True
                current_loop = torch.tensor([])
                break

            deletable_row_position = deletable_row_position[0]
            
            #print('Does my edge to delete contain my edge left index ? ', my_edge[0][0] in edge_to_delete)
            #print()
            actual_new_test = torch.cat((actual_new_test[:deletable_row_position], actual_new_test[deletable_row_position+1:]))
            if(my_edge[0][0] in edge_to_delete):

                deletable_edge_position1 = (actual_new_test_an == edge_to_delete[0]).nonzero(as_tuple=False)
                deletable_edge_position2 = (actual_new_test_an == edge_to_delete[1]).nonzero(as_tuple=False)

                deletable_edge_position1 = deletable_edge_position1[:,0]

                deletable_edge_position2 = deletable_edge_position2[:,0]

                a_cat_b1, counts1 = torch.cat([deletable_edge_position1, deletable_edge_position2]).unique(return_counts=True)
                deletable_row_position = a_cat_b1[torch.where(counts1.gt(1))]
                #print()
                #print('deletable_row_position',deletable_row_position)

                if(deletable_row_position.shape[0]==0):
                    #going_nowhere = True
                    current_loop = torch.tensor([])
                    break

                deletable_row_position = deletable_row_position[0]
                
                actual_new_test_an = torch.cat((actual_new_test_an[:deletable_row_position], actual_new_test_an[deletable_row_position+1:]))    
                actual_new_test = actual_new_test_an
                
            #actual_new_test = torch.cat((actual_new_test[:deletable_row_position], actual_new_test[deletable_row_position+1:]))
            #print('what is this', actual_new_test)
            new_test = actual_new_test

            positions1 = (new_test == left_ind).nonzero(as_tuple=False)
            #print('whats happening here',positions1.shape )
            #print('is the same edge still to delete', edge_to_delete)
            if(tracker ==0):
                break

            '''if(positions1.shape[0]>1):
            edge_to_delete = new_test[positions1[0][0]]
            no_branches_formed = False'''
        else:
            first_position = positions1[0][0]
            #print('first_position',first_position)
            adj_edge1 = new_test[positions1[0][0]]
            other_end1 = abs(positions1 - torch.tensor([[0, 1]]))


            consec_pt1 = new_test[other_end1[0][0]][other_end1[0][1]]
            consec_pt1s = torch.unsqueeze(consec_pt1,0)
            consec_pt_tracker = torch.cat((consec_pt_tracker, consec_pt1s),0)
            consec_pt1 = int(consec_pt1)

                
            #print('consec_pt1',consec_pt1)
            #print('adj_edge1',adj_edge1)
            current_loop = torch.cat((current_loop,adj_edge1),0)
            current_loop1 = current_loop.reshape(int(current_loop.shape[0]/2),2)
            #print('consec_pt_tracker',consec_pt_tracker)
            
            if(consec_pt1 == my_edge[0][0]):
                current_loop = torch.tensor([])
                
            if(consec_pt1 == right_ind):
                my_edge1 = torch.squeeze(my_edge,0)
                current_loop = torch.cat((current_loop,my_edge1),0)
                current_loop1 = current_loop.reshape(int(current_loop.shape[0]/2),2)                
                #found_right_ind = True
                #print('current_loop',current_loop1)
                #current_loop1 = torch.unsqueeze(current_loop1,0)
                #print('current_loop shape now',current_loop1.shape)
                #print('loop_tracker', loop_tracker)
                loops_collec.append(current_loop1)
                #loops_collec[loop_tracker] = current_loop1
                loop_tracker = loop_tracker + 1
                #print()
                #print("Wow! Found a loop here")
                #print()
                #break

            #else:

            left_ind = consec_pt1
            new_test = torch.cat((new_test[:first_position], new_test[first_position+1:]))
            #print('new_test',new_test)
            tracker = tracker+1
    
    #loops_collec = torch.FloatTensor(loops_collec)
    return loops_collec


def get_potential_positions(dist_matrix):
    
    dist_matrix = torch.unique(dist_matrix, dim=0)
    dist_matrix = torch.unique(dist_matrix, dim=1)

    upp_diag = torch.triu(dist_matrix, diagonal=1)

    ff = upp_diag.sort()

    sorted_upper_diag_edges = ff[0]

    sorted_upper_diag_indices = ff[1]

    flattened_uppdg_edges = torch.flatten(sorted_upper_diag_edges)

    non_zero_flattened_uppdg_edges = flattened_uppdg_edges[flattened_uppdg_edges.nonzero()]

    non_zero_flattened_uppdg_edges = non_zero_flattened_uppdg_edges.reshape(non_zero_flattened_uppdg_edges.shape[0])

    increasing_edges = non_zero_flattened_uppdg_edges.sort()[0]
    increasing_edges = torch.unique(increasing_edges, dim=0)
    
    #print('increasing_edges', increasing_edges)
    
    selected_edges = torch.tensor([])
    dead_indices = torch.tensor([])
    potential_triangles = torch.tensor([])
    edge_leads_to_loop = False

    potential_positions = []
    for i in range(increasing_edges.shape[0]):
        a = (upp_diag == increasing_edges[i]).nonzero(as_tuple=False)

        if(selected_edges.shape[0] > 1):
            edge_leads_to_loop = expecting_a_cycle(selected_edges, a)

        if(not(edge_leads_to_loop)):
            selected_edges = torch.cat(((selected_edges, a)), 0)
            
        else:
            potential_positions.append(i)

    #print('selected_edges', selected_edges)
    zeroD_PH = torch.tensor([])
    for i in range(selected_edges.shape[0]):    
        death = dist_matrix[int(selected_edges[i][0])][int(selected_edges[i][1])]
        death = death.reshape(1,1)    
        zeroD_PH = torch.cat(((zeroD_PH, death)), 0)

    births = torch.zeros(zeroD_PH.shape[0], 1)
    zeroD_PH_births_deaths = torch.cat((births, zeroD_PH ),1)
    
    #print(potential_positions)
    return potential_positions


# load trained rAE and bAE
#latent_dims = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2]

latent_dims = [10, 8, 6, 4, 2]

all_hyb_base_models = []

Analys_size = 20

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

    RK_model_reg = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)
    RK_model_base = AE(inp_dim_hyb, hidden_size, latent_dim, no_layers, Sin()).to(device)

    RK_model_reg.load_state_dict(torch.load(path_hyb+'model_regLSTQS'+str(deg_quad)+''+name_hyb, map_location=torch.device('cpu')))
    RK_model_base.load_state_dict(torch.load(path_hyb+'model_baseLSTQS'+str(deg_quad)+''+name_hyb, map_location=torch.device('cpu')))

    #all_hyb_base_models.append(RK_model_reg)

    all_hyb_base_models.append(RK_model_base)

    all_test_coeffs.append(testCoeffs)
    all_X_p.append(X_p)




all_rec_bAE_test = []
for i in range(len(latent_dims)):
    rec_bAE_test = all_hyb_base_models[i].encoder(all_test_coeffs[i].float())#.view(all_test_coeffs[i].shape)
    rec_bAE_test = torch.tensor(rec_bAE_test, requires_grad=False)
    all_rec_bAE_test.append(rec_bAE_test)

#dist_matrix_lat20 = _compute_distance_matrix(testImages, p=2)
dist_matrix_lat20 = _compute_distance_matrix(all_rec_bAE_test[2], p=2)
edges_lat20, edge_lengths_lat20 = get_all_edges(dist_matrix_lat20)

potent_posits = get_potential_positions(dist_matrix_lat20)

#print('all_rec_bAE_test[0].shape', all_rec_bAE_test[8].shape)



print('edges_lat20.shape', edges_lat20.shape)
potential_tracker = 0

components_in_chronology = []

for i in potent_posits:
    potential_tracker = potential_tracker + 1
    edge_collection = edges_lat20[:i+1]

    target_edges = edge_collection[:-1]

    input_edges = edge_collection[:][-1]

    input_edges = input_edges.unsqueeze(0)

    loops = get_all_loops_formed(target_edges, input_edges)

    components_in_chronology.append(loops[0])
    #print('loops'+str(i), loops[0])

    #print('len(loops)', len(loops))

    if(potential_tracker > 13):
        break
#s = 5
print('components_in_chronology', components_in_chronology)

dimsOfComponentsFormed = []

for i in range(len(components_in_chronology)):

    selected_edges = components_in_chronology[i]
    zeroD_PH = torch.tensor([])
    for i in range(selected_edges.shape[0]):    
        death = dist_matrix_lat20[int(selected_edges[i][0])][int(selected_edges[i][1])]
        death = death.reshape(1,1)    
        zeroD_PH = torch.cat(((zeroD_PH, death)), 0)

    births = torch.zeros(zeroD_PH.shape[0], 1)
    zeroD_PH_births_deaths = torch.cat((births, zeroD_PH ),1)

    #print('zeroD_PH_births_deaths', zeroD_PH_births_deaths[:,1])
    dimsOfComponentsFormed.append([selected_edges ,zeroD_PH_births_deaths[:,1]])

print('dimsOfComponentsFormed', dimsOfComponentsFormed)