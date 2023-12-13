'''
HETEROGENEOUS GRAPH TRANSFROMER
We define a heterogeneous graph transformer model to learn node embeddings on the knowledge graph.
'''

# standard imports
import numpy as np
import pandas as pd

# import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# import DGL
import dgl
from dgl.nn.pytorch.conv import HGTConv

# import PyTorch Lightning
import pytorch_lightning as pl

# path manipulation
from pathlib import Path

# import project config file
import sys
sys.path.append('../..')
import project_config

# custom imports
from utils import calculate_metrics

# check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# BILINEAR DECODER CLASS
class BilinearDecoder(pl.LightningModule): # overrides nn.Module

    # INITIALIZATION
    def __init__(self, num_etypes, embedding_dim):
        '''
        This function initializes a bilinear decoder.

        Args:
            num_etypes (int): Number of edge types.
            embedding_dim (int): Dimension of embedding (i.e., output dimension * number of attention heads).
        '''
        super().__init__()

        # edge-type specific learnable weights
        self.relation_weights = nn.Parameter(torch.Tensor(num_etypes, embedding_dim))

        # initialize weights
        nn.init.xavier_uniform_(self.relation_weights, gain = nn.init.calculate_gain('leaky_relu'))
    

    # ADD EDGE TYPE INDEX
    def add_edge_type_index(self, edge_graph):
        '''
        This function adds an integer edge type label to each edge in the graph. This is required for the decoder.
        Specifically, the edge type label is used to subset the right row of the relation weight matrix.
        
        Args:
            edge_graph (dgl.DGLGraph): Positive or negative edge graph.
        '''

        # iterate over the canonical edge types
        for edge_index, edge_type in enumerate(edge_graph.canonical_etypes):
        
            # get number of edges of that type
            num_edges = edge_graph.num_edges(edge_type)

            # add integer label to edge
            edge_graph.edges[edge_type].data['edge_type_index'] = torch.tensor([edge_index] * num_edges, device = self.device) #.to(device)
    
    
    # DECODER
    def decode(self, edges):
        '''
        This is a user-defined function over the edges to generate the score for each edge.
        See https://docs.dgl.ai/en/0.9.x/generated/dgl.DGLGraph.apply_edges.html.
        '''
        
        # get source embeddings
        src_embeddings = edges.src['node_embedding']
        dst_embeddings = edges.dst['node_embedding']

        # apply activation function
        src_embeddings = F.leaky_relu(src_embeddings)
        dst_embeddings = F.leaky_relu(dst_embeddings)

        # get relation weight for specific edge type
        # note that, because the decode function is applied by edge type, we can use the first edge to get the edge type
        edge_type_index = edges.data['edge_type_index'][0] # see torch.unique(edges.data['edge_type'])
        rel_weights = self.relation_weights[edge_type_index]

        # compute weighted dot product
        # each row of src_embeddings is multiplied by rel_weights, then element-wise multiplied by dst_embeddings
        # finally, a row-wise sum is performed to get a single score per edge
        score = torch.sum(src_embeddings * rel_weights * dst_embeddings, dim = 1)

        return {'score': score}


    # COMPUTE SCORE
    def compute_score(self, edge_graph):
        '''
        This function computes the score for positive or negative edges using dgl.DGLGraph.apply_edges.

        Args:
            edge_graph (dgl.DGLGraph): Positive or negative edge graph.
        '''

        with edge_graph.local_scope():

            # get edge types with > 0 number of edges in the positive graph
            nonzero_edge_types = [etype for etype in edge_graph.canonical_etypes if edge_graph.num_edges(etype) != 0]

            # compute score for positive graph
            for etype in nonzero_edge_types:
                edge_graph.apply_edges(self.decode, etype = etype)
            
            # return scores
            return edge_graph.edata['score']
    
    
    # FORWARD PASS
    def forward(self, subgraph, pos_graph, neg_graph, node_embeddings):
        '''
        This function performs a forward pass of the bilinear decoder.

        Args:
            subgraph (dgl.DGLHeteroGraph): Subgraph.
            pos_graph (dgl.DGLHeteroGraph): Positive graph.
            neg_graph (dgl.DGLHeteroGraph): Negative graph.
            node_embeddings (torch.Tensor): Node embeddings.
        '''

        # get subgraph node IDs
        subgraph_nodes = subgraph.ndata['node_index']

        # assign node embeddings to positive and negative graphs
        # iterate over node types in positive graph
        for ntype in pos_graph.ntypes:

            # get positive graph node IDs
            pos_graph_nodes = pos_graph.ndata['node_index'][ntype].unsqueeze(1)
            
            # find indices of positive graph nodes in subgraph
            # note, that indices are same for negative graph
            # compare pos_graph.ndata['_ID'] vs. neg_graph.ndata['_ID']
            pos_graph_indices = torch.where(subgraph_nodes == pos_graph_nodes)[1]

            # add embeddings as feature to graph
            pos_graph.nodes[ntype].data['node_embedding'] = node_embeddings[pos_graph_indices]
            neg_graph.nodes[ntype].data['node_embedding'] = node_embeddings[pos_graph_indices]

        # add edge indices to positive and negative graphs
        self.add_edge_type_index(pos_graph)
        self.add_edge_type_index(neg_graph)

        # compute scores for positive and negative graphs
        pos_graph_scores = self.compute_score(pos_graph)
        neg_graph_scores = self.compute_score(neg_graph)

        # return scores
        return pos_graph_scores, neg_graph_scores
    

# HETEROGENEOUS GRAPH TRANSFORMER
class HGT(pl.LightningModule):
    
    # INITIALIZATION
    def __init__(self, num_nodes, num_ntypes, num_etypes, num_feat = 1024, num_heads = 4,
                 hidden_dim = 256, output_dim = 128, num_layers = 2,
                 dropout_prob = 0.5, pred_threshold = 0.5,
                 lr = 0.0001, wd = 0.0, lr_factor = 0.01, lr_patience = 100, lr_threshold = 1e-4,
                 lr_threshold_mode = 'rel', lr_cooldown = 0, min_lr = 1e-8, eps = 1e-8,
                 hparams = None):
        '''
        This function initializes the model and defines the model hyperparameters and architecture.

        Args:
            num_nodes (int): Number of nodes in the graph.
            num_ntypes (int): Number of node types in the graph.
            num_etypes (int): Number of edge types in the graph.
            num_feat (int): Number of input features (i.e., hidden embedding dimension).
            num_heads (int): Number of attention heads.
            hidden_dim (int): Number of hidden units in the second to last HGT layer.
            output_dim (int): Number of output units.
            num_layers (int): Number of HGT layers.
            dropout_prob (float): Dropout probability.
            pred_threshold (float): Prediction threshold to compute metrics.
            lr (float): Learning rate.
            wd (float): Weight decay.
            lr_factor (float): Factor by which to reduce learning rate.
            lr_patience (int): Number of epochs with no improvement after which learning rate will be reduced.
            lr_threshold (float): Threshold for measuring the new optimum, to only focus on significant changes.
            lr_threshold_mode (str): One of ['rel', 'abs'].
            lr_cooldown (int): Number of epochs to wait before resuming normal operation after lr reduction.
            min_lr (float): A lower bound on the learning rate of all param groups or each group respectively.
            eps (float): Term added to the denominator to improve numerical stability.
            hparams (dict): Dictionary of model hyperparameters. Will override all other arguments if not None.
        '''

        super().__init__()

        # if hparams_dict is None, construct dictionary from arguments
        if hparams is None:
            hparams = locals()

        # save model hyperparameters
        self.save_hyperparameters(hparams)
        self.num_feat = hparams['num_feat']
        self.num_heads = hparams['num_heads']
        self.hidden_dim = hparams['hidden_dim']
        self.output_dim = hparams['output_dim']
        self.num_layers = hparams['num_layers']
        self.dropout_prob = hparams['dropout_prob']
        self.pred_threshold = hparams['pred_threshold']

        # learning rate parameters
        self.lr = hparams['lr']
        self.wd = hparams['wd']
        self.lr_factor = hparams['lr_factor']
        self.lr_patience = hparams['lr_patience']
        self.lr_threshold = hparams['lr_threshold']
        self.lr_threshold_mode = hparams['lr_threshold_mode']
        self.lr_cooldown = hparams['lr_cooldown']
        self.min_lr = hparams['min_lr']
        self.eps = hparams['eps']

        # calculate sizes of hidden dimensions
        self.h_dim_1 = hidden_dim * 2
        self.h_dim_2 = hidden_dim

        # define node embeddings
        self.emb = nn.Embedding(num_nodes, num_feat)

        # layer 1
        self.conv1 = HGTConv(in_size = num_feat, head_size = self.h_dim_1, num_heads = num_heads,
                                num_ntypes = num_ntypes, num_etypes = num_etypes, dropout = 0.2, use_norm = True)
        
        # layer normalization 1
        self.norm1 = nn.LayerNorm(self.h_dim_1 * num_heads)

        if self.num_layers == 2:
        
            # layer 2
            self.conv2 = HGTConv(in_size = self.h_dim_1 * num_heads, head_size = output_dim, num_heads = num_heads,
                                    num_ntypes = num_ntypes, num_etypes = num_etypes, dropout = 0.2, use_norm = True)
            
        elif self.num_layers == 3:
        
            # layer 2
            self.conv2 = HGTConv(in_size = self.h_dim_1 * num_heads, head_size = self.h_dim_2, num_heads = num_heads,
                                    num_ntypes = num_ntypes, num_etypes = num_etypes, dropout = 0.2, use_norm = True)

            # layer normalization 2
            self.norm2 = nn.LayerNorm(self.h_dim_2 * num_heads)

            # layer 3
            self.conv3 = HGTConv(in_size = self.h_dim_2 * num_heads, head_size = output_dim, num_heads = num_heads,
                                    num_ntypes = num_ntypes, num_etypes = num_etypes, dropout = 0.2, use_norm = True)
            
        else:

            # raise error
            raise ValueError('Number of layers must be 2 or 3.')

        # define decoder
        self.decoder = BilinearDecoder(num_etypes, output_dim * num_heads)
    
    
    # FORWARD PASS
    def forward(self, subgraph):
        '''
        This function performs a forward pass of the model. Note that the subgraph must be converted to from a 
        heterogeneous graph to homogeneous graph for efficiency.

        Args:
            subgraph (dgl.DGLHeteroGraph): Subgraph containing the nodes and edges for the current batch.
        '''

        # get global indices
        global_node_indices = subgraph.ndata['node_index']

        # get node embeddings from the first MFG layer
        x = self.emb(global_node_indices)

        # pass node embedding through first two layers
        x = self.conv1(subgraph, x, subgraph.ndata[dgl.NTYPE], subgraph.edata[dgl.ETYPE])
        x = self.norm1(x)
        x = F.leaky_relu(x)
        x = self.conv2(subgraph, x, subgraph.ndata[dgl.NTYPE], subgraph.edata[dgl.ETYPE])

        # check if 3 layers
        if self.num_layers == 3:

            # pass node embedding through layer 3
            x = self.norm2(x)
            x = F.leaky_relu(x)
            x = self.conv3(subgraph, x, subgraph.ndata[dgl.NTYPE], subgraph.edata[dgl.ETYPE])

        # return node embeddings
        return x
    

    # STEP FUNCTION USED FOR TRAINING, VALIDATION, AND TESTING
    def _step(self, input_nodes, pos_graph, neg_graph, subgraph, mode):
        '''Defines the step that is run on each batch of data. PyTorch Lightning handles steps including:
            - Moving data to the correct device.
            - Epoch and batch iteration.
            - optimizer.step(), loss.backward(), optimizer.zero_grad() calls.
            - Calling of model.eval(), enabling/disabling grads during evaluation.
            - Logging of metrics.
        
        Args:
            input_nodes (torch.Tensor): Input nodes.
            pos_graph (dgl.DGLHeteroGraph): Positive graph.
            neg_graph (dgl.DGLHeteroGraph): Negative graph.
            subgraph (dgl.DGLHeteroGraph): Subgraph.
            mode (str): The mode of the step (train, val, test).
        '''

        # get batch size by summing number of nodes in each node type
        batch_size = sum([x.shape[0] for x in input_nodes.values()])

        # convert heterogeneous graph to homogeneous graph for efficiency
        # see https://docs.dgl.ai/en/latest/generated/dgl.to_homogeneous.html
        subgraph = dgl.to_homogeneous(subgraph, ndata = ['node_index'])
        
        # send to GPU
        # subgraph = subgraph.to(device)
        # pos_graph = pos_graph.to(device)
        # neg_graph = neg_graph.to(device)

        # get node embeddings
        node_embeddings = self.forward(subgraph)

        # compute score from decoder
        pos_scores, neg_scores = self.decoder(subgraph, pos_graph, neg_graph, node_embeddings)

        # compute loss
        loss, metrics = self.compute_loss(pos_scores, neg_scores)

        # return loss and metrics
        return loss, metrics, batch_size
    

    # TRAINING STEP
    def training_step(self, batch, batch_idx):
        '''Defines the step that is run on each batch of training data.'''

        # get batch elements
        input_nodes, pos_graph, neg_graph, subgraph = batch

        # get loss and metrics
        loss, metrics, batch_size = self._step(input_nodes, pos_graph, neg_graph, subgraph, mode = 'train')

        # log loss and metrics
        values = {"train/loss": loss.detach(),
                  "train/accuracy": metrics['accuracy'],
                  "train/ap": metrics['ap'],
                  "train/f1": metrics['f1'],
                  "train/auroc": metrics['auroc']}
        self.log_dict(values, batch_size = batch_size)

        # return loss
        return loss
    

    # VALIDATION STEP
    def validation_step(self, batch, batch_idx):
        '''Defines the step that is run on each batch of validation data.'''

        # get batch elements
        input_nodes, pos_graph, neg_graph, subgraph = batch

        # get loss and metrics
        loss, metrics, batch_size = self._step(input_nodes, pos_graph, neg_graph, subgraph, mode = 'val')

        # log loss and metrics
        values = {"val/loss": loss.detach(),
                  "val/accuracy": metrics['accuracy'],
                  "val/ap": metrics['ap'],
                  "val/f1": metrics['f1'],
                  "val/auroc": metrics['auroc']}
        self.log_dict(values, batch_size = batch_size)


    # TEST STEP
    def test_step(self, batch, batch_idx):
        '''Defines the step that is run on each batch of test data.'''

        # get batch elements
        input_nodes, pos_graph, neg_graph, subgraph = batch

        # get loss and metrics
        loss, metrics, batch_size = self._step(input_nodes, pos_graph, neg_graph, subgraph, mode = 'test')

        # log loss and metrics
        values = {"test/loss": loss.detach(),
                  "test/accuracy": metrics['accuracy'],
                  "test/ap": metrics['ap'],
                  "test/f1": metrics['f1'],
                  "test/auroc": metrics['auroc']}
        self.log_dict(values, batch_size = batch_size)

    
    # LOSS FUNCTION
    def compute_loss(self, pos_scores, neg_scores):
        '''
        This function computes the loss and metrics for the current batch.
        '''

        # concatenate positive and negative scores across edge types
        pos_pred = torch.cat(list(pos_scores.values()))
        neg_pred = torch.cat(list(neg_scores.values()))
        raw_pred = torch.cat((pos_pred, neg_pred))

        # transform with activation function
        pred = torch.sigmoid(raw_pred)

        # construct target vector
        pos_target = torch.ones(pos_pred.shape[0])
        neg_target = torch.zeros(neg_pred.shape[0])
        target = torch.cat((pos_target, neg_target)).to(self.device) #.to(device)

        # compute loss
        loss = F.binary_cross_entropy(pred, target, reduction = "mean")

        # calculate metrics
        metrics = calculate_metrics(pred.cpu().detach().numpy(), target.cpu().detach().numpy(), self.pred_threshold)
        return loss, metrics
    

    # OPTIMIZER AND SCHEDULER
    def configure_optimizers(self):
        '''
        This function is called by PyTorch Lightning to get the optimizer and scheduler.
        We reduce the learning rate by a factor of lr_factor if the validation loss does not improve for lr_patience epochs.

        Args:
            None

        Returns:
            dict: Dictionary containing the optimizer and scheduler.
        '''
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode = 'min', factor = self.lr_factor, patience = self.lr_patience,
            threshold = self.lr_threshold, threshold_mode = self.lr_threshold_mode,
            cooldown = self.lr_cooldown, min_lr = self.min_lr, eps = self.eps
        )
        
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    'name': 'curr_lr'
                    },
                }