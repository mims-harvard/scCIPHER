'''
HYPERPARAMETERS

This file contains the hyperparameters for the node embedder.
'''

# argument parser for command line arguments
import argparse

# import project configuration file
import sys
sys.path.append('../..')
import project_config


# COMMAND LINE ARGUMENTS
def parse_args():
    '''
    Parse command line arguments.
    '''

    parser = argparse.ArgumentParser(description = "Learn node embeddings.")

    # input and output file paths
    parser.add_argument('--node_list', type = str, default = project_config.KG_DIR / 'neuroKG_nodes.csv', help = 'Path to node list.')
    parser.add_argument('--edge_list', type = str, default = project_config.KG_DIR / 'neuroKG_edges.csv', help = 'Path to edge list.')
    parser.add_argument('--save_dir', type = str, default = project_config.RESULTS_DIR / 'pretrain', help = 'Directory for saving files.')
    
    # tunable parameters
    parser.add_argument('--num_feat', type = int, default = 2048, help = 'Dimension of embedding layer.')
    parser.add_argument('--num_heads', default = 4, type = int)
    parser.add_argument('--hidden_dim', default = 128, type = int)
    parser.add_argument('--output_dim', default = 64, type = int)
    parser.add_argument('--wd', default = 0.0, type = float)
    parser.add_argument('--dropout_prob', type = float, default = 0.3, help = 'Dropout probability.')
    parser.add_argument('--lr', default = 0.0001, type = float)
    parser.add_argument('--max_epochs', default = 250, type = int)
    
    # resume with best checkpoint
    parser.add_argument('--resume', default = "", type = str)
    parser.add_argument('--best_ckpt', type = str, default = None, help = 'Name of the best performing checkpoint.')
    
    # save embeddings and debugging modes
    parser.add_argument('--save_embeddings', action = 'store_true') # include --save_embeddings flag to save embeddings
    parser.add_argument('--debug', action = 'store_true') # include --debug flag to debug

    args = parser.parse_args()
    return args


# PRE-TRAINING HYPERPARAMETERS
def get_hyperparameters(args):   
    '''
    Return hyperparameters for node embedder. Combine tunable hyperparameters with fixed hyperparameters.
    See parse_args() for all possible command line arguments.

    Args:
        args: command line arguments
    
    Tunable Parameters:
        num_feat: dimension of embedding layer
        num_heads: number of attention heads
        hidden_dim: dimension of hidden layer
        output_dim: dimension of output layer
        wd: weight decay
        dropout: dropout probability
        lr: learning rate
    '''

    # generate dictionary from command-line arguments
    args_dict = vars(args)

    # define fanout
    fanout = [1, 1, 1] #[1, 1]

    # default hyperparameters
    hparams_dict = {
                # fixed parameters
                'pred_threshold': 0.5,
                'n_gpus': 1,
                'num_workers': 8,
                'train_batch_size': 768,
                'val_batch_size': 768,
                'test_batch_size': 768,
                'sampler_fanout': fanout,
                'num_layers': len(fanout),
                'negative_k': 1,
                'grad_clip': 1.0,
                'lr_factor': 0.01,
                'lr_patience': 100,
                'lr_threshold': 1e-4,
                'lr_threshold_mode': 'rel',
                'lr_cooldown': 0,
                'min_lr': 0,
                'eps': 1e-8,
                'seed': 42,
                'profiler': None,
                # see https://github.com/wandb/wandb/issues/714
                'wandb_save_dir': project_config.RESULTS_DIR / 'wandb' / 'pretrain',
                'log_every_n_steps': 10,
                'time': False,
                'sample_subgraph': False,
                'seed_node': 1,
                'n_walks': 100,
                'walk_length': 10
        }
    
    # combine tunable hyperparameters with fixed hyperparameters
    hparams = dict(args_dict, **hparams_dict)
    
    print('Pre-Training Hyperparameters: ', hparams)
    
    return hparams
