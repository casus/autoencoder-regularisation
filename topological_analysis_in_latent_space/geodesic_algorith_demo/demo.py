#from pickle import TRUE
import sys
sys.path.append('./')

import torch

import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline

import torch
device = torch.device('cpu')


import matplotlib
matplotlib.rcdefaults() 


# All Functions 
def _compute_distance_matrix(x, p=2):
    x_flat = x.view(x.size(0), -1)

    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)

    return distances


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


def get_all_edges(dist_matrix_):
    
    dist_matrix = torch.unique(dist_matrix_, dim=0)
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
    
    
    selected_edges = torch.tensor([])
    dead_indices = torch.tensor([])
    potential_triangles = torch.tensor([])
    edge_leads_to_loop = False

    for i in range(increasing_edges.shape[0]):
        a = (dist_matrix_ == increasing_edges[i]).nonzero(as_tuple=False)
        if(selected_edges.shape[0] > 1):
            edge_leads_to_loop = False #expecting_a_cycle(selected_edges, a)

        if(not(edge_leads_to_loop)):
            selected_edges = torch.cat(((selected_edges, a[0].unsqueeze(0))), 0)

    return selected_edges, increasing_edges

    
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
        
        if(positions1.shape[0]>1):
            other_end_con = abs(positions1 - torch.tensor([[0, 1]]))
            consec_pt_con = new_test[other_end_con[0][0]][other_end_con[0][1]]

            
            if(not(other_side_unbranched_pts.shape[0] == 0)):
                if(not(consec_pt_con in consec_pt_tracker) and not(consec_pt_con==other_side_unbranched_pts[0])):
                    edge_to_delete = new_test[positions1[0][0]]
            else:
                if(not(consec_pt_con in consec_pt_tracker)):
                    edge_to_delete = new_test[positions1[0][0]]                
            no_branches_formed = False
        branches_rising = positions1.shape[0]

        if(positions1.shape[0]==0):
            current_loop = torch.tensor([])
            consec_pt_tracker = torch.tensor([])

            left_ind = my_edge[0][0]

            deletable_edge_position1 = (actual_new_test == edge_to_delete[0]).nonzero(as_tuple=False)
            deletable_edge_position2 = (actual_new_test == edge_to_delete[1]).nonzero(as_tuple=False)

            deletable_edge_position1 = deletable_edge_position1[:,0]

            deletable_edge_position2 = deletable_edge_position2[:,0]

            a_cat_b1, counts1 = torch.cat([deletable_edge_position1, deletable_edge_position2]).unique(return_counts=True)
            deletable_row_position = a_cat_b1[torch.where(counts1.gt(1))]

            
            if(deletable_row_position.shape[0]==0):
                current_loop = torch.tensor([])
                break

            deletable_row_position = deletable_row_position[0]

            actual_new_test = torch.cat((actual_new_test[:deletable_row_position], actual_new_test[deletable_row_position+1:]))
            if(my_edge[0][0] in edge_to_delete):

                deletable_edge_position1 = (actual_new_test_an == edge_to_delete[0]).nonzero(as_tuple=False)
                deletable_edge_position2 = (actual_new_test_an == edge_to_delete[1]).nonzero(as_tuple=False)

                deletable_edge_position1 = deletable_edge_position1[:,0]

                deletable_edge_position2 = deletable_edge_position2[:,0]

                a_cat_b1, counts1 = torch.cat([deletable_edge_position1, deletable_edge_position2]).unique(return_counts=True)
                deletable_row_position = a_cat_b1[torch.where(counts1.gt(1))]

                if(deletable_row_position.shape[0]==0):
                    current_loop = torch.tensor([])
                    break

                deletable_row_position = deletable_row_position[0]
                
                actual_new_test_an = torch.cat((actual_new_test_an[:deletable_row_position], actual_new_test_an[deletable_row_position+1:]))    
                actual_new_test = actual_new_test_an
                

            new_test = actual_new_test

            positions1 = (new_test == left_ind).nonzero(as_tuple=False)

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
    
    return loops_collec


def get_potential_positions(input_edges, all_edges):
    
    potential_positions = []
    for i in range(all_edges.shape[0]-1):

        edge_leads_to_loop = expecting_a_cycle( all_edges[:i+1], input_edges)

        if(edge_leads_to_loop):
            potential_positions.append(i)        

        if (len(potential_positions) == 1):
            break

    return potential_positions



# Indices of images between which we want to compute the approximation of geodeic
input_im1 = 8
input_im2 = 83


#################################################################################################################################################
# Get datase Download the dataset
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

Analys_size = 100

# Load the test data set as torch tensors
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)
for i, data in enumerate(testloader, 0):
    inputs, _ = data
    inputs = inputs.numpy()
    inputs = torch.tensor(inputs)
    break


testImages = inputs[:Analys_size]
#################################################################################################################################################

def compute_distance_matrix_sliced_wasserstein(input_point_cloud):
    wass_dist_matrix = np.zeros((input_point_cloud.shape[0], input_point_cloud.shape[0]))
    for i in range(input_point_cloud.shape[0]):
        for j in range(input_point_cloud.shape[0]):
            if(i==j):
                wass_dist_matrix[j, i] = 0
            else:
                wassDistance = ot.sliced_wasserstein_distance(input_point_cloud[i][0], input_point_cloud[j][0], seed=0)  

                wass_dist_matrix[j, i] = wassDistance                
    wass_dist_matrix = torch.tensor(wass_dist_matrix)
    return wass_dist_matrix


dist_matrix_lat20 = _compute_distance_matrix(testImages, p=2)


edges_lat20, edge_lengths_lat20 = get_all_edges(dist_matrix_lat20)


input_edges = torch.tensor([[input_im1, input_im2]])

potent_posits = get_potential_positions(input_edges, edges_lat20)

potential_tracker = 0

components_in_chronology = []

for i in potent_posits:
    potential_tracker = potential_tracker + 1
    edge_collection = edges_lat20[:i+1]

    target_edges = edge_collection

    loops = get_all_loops_formed(target_edges, input_edges)

    

    print('loops', loops[0])

    print('len(loops)', len(loops))

    if(potential_tracker==0):
        break


flat_geod4_base = loops[0].flatten()

printed = []
for i in range(len(flat_geod4_base)):
    if (flat_geod4_base[i] not in printed):
        print(flat_geod4_base[i])
        printed.append(flat_geod4_base[i])

print('printed',printed)


Geodesic = [int(a) for a in printed]

print("A", Geodesic)
t = 0

wassDistList = []
for k in Geodesic:

    print(k)
    plt.imshow(testImages[k][0])
    plt.axis('off')
    plt.savefig('./topological_analysis_in_latent_space/FashionMNIST_geodesics/saved_geo/gamma'+str(k)+'value.png')

