'''
CONSTRUCT DATA LOADERS
This script loads the knowledge graph into a Deep Graph Library (DGL) object and constructs the data loaders 
for the node embedding model (e.g., batched data loaders, neighbor samplers, etc.).
'''

# standard imports
import numpy as np
import pandas as pd

# import PyTorch and DGL
import torch
import dgl

# path manipulation
from pathlib import Path

# import project configuration file
import sys
sys.path.append('../..')
import project_config

# custom imports
from utils import generate_subgraph
from samplers import FixedSampler

# check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# LOAD KNOWLEDGE GRAPH
def load_graph(hparams):

    # read in nodes and edges
    # could also provide as args.node_list and args.edge_list with arg as argument of function
    nodes = pd.read_csv(hparams['node_list'], dtype = {'node_index': int}, low_memory = False)
    edges = pd.read_csv(hparams['edge_list'], dtype = {'edge_index': int, 'x_index': int, 'y_index': int}, low_memory = False)

    # if sample subgraph, subsample nodes and edges
    if hparams['sample_subgraph']:
        nodes, edges = generate_subgraph(nodes, edges, hparams['seed_node'], hparams['n_walks'], hparams['walk_length'])
        print("Number of subgraph nodes: ", len(nodes))
        print("Number of subgraph edges: ", len(edges))

        # save subgraph
        nodes.to_csv(Path(hparams['save_dir']) / 'subgraph_nodes.csv', index = False)
        edges.to_csv(Path(hparams['save_dir']) / 'subgraph_edges.csv', index = False)

    # group the nodes DataFrame by 'node_type' and use cumcount to generate the 'node_type_index'
    nodes['node_type_index'] = nodes.groupby('node_type').cumcount()

    # use the 'node_type_index' column to create the 'x_type_index' and 'y_type_index' columns in the edges DataFrame
    edges['x_type_index'] = nodes.loc[edges['x_index'], 'node_type_index'].values
    edges['y_type_index'] = nodes.loc[edges['y_index'], 'node_type_index'].values

    # define empty dictionary to store graph data
    neuroKG_data = {}

    # group the edges DataFrame by unique combinations of x_type, relation, and y_type
    grouped_edges = edges.groupby(['x_type', 'relation', 'y_type'], sort = False)

    # iterate over the groups
    for (x_type, relation, y_type), edges_subset in grouped_edges:

        # convert edge indices to torch tensor
        edge_indices = (torch.tensor(edges_subset['x_type_index'].values), torch.tensor(edges_subset['y_type_index'].values))

        # add edge indices to data object
        neuroKG_data[(x_type, relation, y_type)] = edge_indices

        # print update
        # print(f'Added edge relation: {x_type} - {relation} - {y_type}')

    # instantiate a DGL HeteroGraph
    neuroKG = dgl.heterograph(neuroKG_data)

    # add node features to the heterograph
    # first, group the nodes DataFrame by node_type
    grouped_nodes = nodes.groupby('node_type', sort = False)

    # iterate over the groups and add global node indices to the graph
    for node_type, nodes_subset in grouped_nodes:

        neuroKG.nodes[node_type].data['node_index'] = torch.tensor(nodes_subset['node_index'].values)

    # return the graph
    return neuroKG


# PARTITION GRAPH
def partition_graph(neuroKG, hparams):

    # define dictionaries for train, validation, and test eids
    train_eids = {}
    val_eids = {}
    test_eids = {}

    # split the edges into train, validation, and test sets
    forward_edge_types = [x for x in neuroKG.canonical_etypes if "rev" not in x[1]]
    for etype in forward_edge_types:

        # subset edge IDs for the current edge type
        etype_eids = neuroKG.edges(etype = etype, form = 'eid')

        # randomly shuffle edge IDs
        num_edges = etype_eids.shape[0]
        type_eids = etype_eids[torch.randperm(num_edges)]

        # get train, validation, and test lengths
        # here, we use a 80/15/5 split
        test_length = int(np.ceil(0.05 * num_edges))
        val_length = int(np.ceil(0.15 * num_edges))
        train_length = num_edges - test_length - val_length

        # split the edge IDs into train, validation, and test sets
        etype_train_eids = etype_eids[:train_length]
        etype_val_eids = etype_eids[train_length:(train_length + val_length)]
        etype_test_eids = etype_eids[(train_length + val_length):]

        # print number of edges in each set
        # print("Edges of type {} split into {} train, {} validation, and {} test edges.".format(etype, len(etype_train_eids), len(etype_val_eids), len(etype_test_eids)))

        # add the edge IDs to the dictionaries
        train_eids[etype] = etype_train_eids
        val_eids[etype] = etype_val_eids
        test_eids[etype] = etype_test_eids

        # get reverse edge type
        reverse_etype = (etype[2], "rev_" + etype[1], etype[0])

        # add the reverse edge IDs to the dictionaries
        train_eids[reverse_etype] = etype_train_eids
        val_eids[reverse_etype] = etype_val_eids
        test_eids[reverse_etype] = etype_test_eids
    
    # define new training graph
    train_neuroKG = neuroKG.edge_subgraph(train_eids, relabel_nodes = False)

    # combine train and validation edge IDs
    train_val_eids = {}
    for etype in train_eids.keys():
        train_val_eids[etype] = torch.cat((train_eids[etype], val_eids[etype]))

    # define new validation graph
    val_neuroKG = neuroKG.edge_subgraph(train_val_eids, relabel_nodes = False)

    # define new test graph
    test_neuroKG = neuroKG.edge_subgraph(test_eids, relabel_nodes = False)

    # return the graphs
    return train_neuroKG, val_neuroKG, test_neuroKG, train_eids, val_eids, test_eids


# CREATE DATA LOADERS
def create_dataloaders(neuroKG, train_neuroKG, val_neuroKG, test_neuroKG,
                       train_eids, val_eids, test_eids,
                       sampler_fanout = [1, 1, 1], negative_k = 8,
                       train_batch_size = 8, val_batch_size = 8, test_batch_size = 8,
                       num_workers = 0):

    print('Creating mini-batch pre-training dataloader...')

    # define dictionary mapping forward edges to reverse edges, and vice versa
    forward_edge_types = [x for x in neuroKG.canonical_etypes if "rev" not in x[1]]
    reverse_edge_dict = {(u, r, v): (v, "rev_" + r, u) for u, r, v in forward_edge_types}
    reverse_edge_dict.update({value: key for key, value in reverse_edge_dict.items()})
    
    # define positive sampler
    sampler = FixedSampler(sampler_fanout, fixed_k = 10, upsample_rare_types = True)

    # other choices for positive sampler
    # see https://docs.dgl.ai/en/latest/generated/dgl.dataloading.as_edge_prediction_sampler.html
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3) # 3-layer full neighbor sampler
    # sampler = dgl.dataloading.NeighborSampler([1, 1, 1]) # requires blocks
    # sampler = dgl.dataloading.ShaDowKHopSampler(sampler_fanout)

    # define negative sampler
    # generate 5 negative samples per edge using uniform distribution
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(negative_k)
    # define reverse edge types for each positive edge type and vice versa

    # convert to edge sampler
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        exclude = "reverse_types", # exclude reverse edges
        reverse_etypes = reverse_edge_dict, # define reverse edge types
        negative_sampler = neg_sampler)

    # define training dataloader
    train_dataloader = dgl.dataloading.DataLoader(
        train_neuroKG, train_eids, sampler,
        batch_size = train_batch_size,
        shuffle = True,
        drop_last = False,
        num_workers = num_workers)

    # define validation dataloader
    val_dataloader = dgl.dataloading.DataLoader(
        val_neuroKG, val_eids, sampler,
        batch_size = val_batch_size,
        shuffle = True,
        drop_last = False,
        num_workers = num_workers)

    # define test dataloader
    test_dataloader = dgl.dataloading.DataLoader(
        test_neuroKG, test_eids, sampler,
        batch_size = test_batch_size,
        shuffle = True,
        drop_last = False,
        num_workers = num_workers)
    
    # return the dataloaders
    return train_dataloader, val_dataloader, test_dataloader