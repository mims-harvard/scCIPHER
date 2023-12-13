'''
PRETRAIN NODE EMBEDDING MODEL
This script contains the main function for pretraining the node embedding model.
'''

# standard imports
import numpy as np
import pandas as pd
from datetime import datetime

# import PyTorch and DGL
import torch
import dgl

# import PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# path manipulation
from pathlib import Path

# import project configuration file
import sys
sys.path.append('../..')
import project_config

# custom imports
from hyperparameters import parse_args, get_hyperparameters
from dataloaders import load_graph, partition_graph, create_dataloaders
from models import HGT

# check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# PRE-TRAINING FUNCTION
def pretrain(hparams):
    '''
    Pre-train node embedding model.

    Args:
        hparams: hyperparameters for node embedding model
    '''

    # set seed
    pl.seed_everything(hparams['seed'], workers = True)

    # load NeuroKG knowledge graph
    neuroKG = load_graph(hparams)

    # partition graph into train, validation, and test sets
    train_neuroKG, val_neuroKG, test_neuroKG, train_eids, val_eids, test_eids = partition_graph(neuroKG, hparams)

    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        neuroKG, train_neuroKG, val_neuroKG, test_neuroKG, train_eids, val_eids, test_eids,
        sampler_fanout = hparams['sampler_fanout'], negative_k = hparams['negative_k'],
        train_batch_size = hparams['train_batch_size'], val_batch_size = hparams['val_batch_size'],
        test_batch_size = hparams['test_batch_size'], num_workers = hparams['num_workers']
    )

    # enable CPU affinity
    train_dataloader.enable_cpu_affinity()
    val_dataloader.enable_cpu_affinity()
    test_dataloader.enable_cpu_affinity()

    # instantiate logger
    curr_time = datetime.now()
    run_name = curr_time.strftime('%H:%M:%S on %m/%d/%Y')
    run_id = curr_time.strftime('%Y_%m_%d_%H_%M_%S')
    wandb_logger = WandbLogger(name = run_name, project = 'cipher-pretraining', entity = 'ayushnoori',
                               save_dir = hparams['wandb_save_dir'], id = run_id, resume = "allow")

    # instantiate model
    model = HGT(
        num_nodes = train_neuroKG.num_nodes(), num_ntypes = len(train_neuroKG.ntypes),
        num_etypes = len(train_neuroKG.canonical_etypes), hparams = hparams
    )

    # define callbacks
    checkpoint_callback = ModelCheckpoint(monitor = 'val/auroc', 
                                          dirpath = Path(hparams['save_dir']) / 'checkpoints',
                                          filename = f'{run_id}' + '_{epoch}-{step}',
                                          save_top_k = 1, mode = 'max')
    lr_monitor = LearningRateMonitor(logging_interval = 'step')
    wandb_logger.watch(model, log = 'all')

    # set debugging mode
    if hparams['debug']:
        limit_train_batches = 100
        limit_val_batches = 5 
        hparams['max_epochs'] = 3
        hparams['log_every_n_steps'] = 1
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0 

    # define trainer
    trainer = pl.Trainer(
        devices = 1 if torch.cuda.is_available() else 0,
        accelerator = "gpu", # "auto",
        logger = wandb_logger,
        max_epochs = hparams['max_epochs'],
        callbacks=[checkpoint_callback, lr_monitor], 
        gradient_clip_val = hparams['grad_clip'],
        profiler = hparams['profiler'],
        log_every_n_steps = hparams['log_every_n_steps'],
        # limit_train_batches = limit_train_batches,
        # limit_val_batches = limit_val_batches,
        val_check_interval=0.25, # check validation set 4 times per epoch
        deterministic = True,
    )

    # pre-train model
    trainer.fit(model, train_dataloader, val_dataloader)

    # test model
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    
    # get hyperparameters
    args = parse_args()
    hparams = get_hyperparameters(args) 

    # after training is complete, save node embeddings
    if args.save_embeddings:
        # save node embeddings from a trained model
        # save_embeddings(args, hparams)
        pass
    else:
        # train model
        pretrain(hparams)