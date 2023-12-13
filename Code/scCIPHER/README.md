# scCIPHER Pre-Training

This directory contains the code to pretrain the node embedding model.

⏳ Note that the code for the node embedding model is still under development.

Key files included in this directory are as follows:

- ⚙️ `hyperparameters.py`: This script contains the hyperparameters for the node embedding model, including the argument parser to read command line arguments to `pretrain.py`.

- 💾 `dataloaders.py`: This script  loads the knowledge graph into a Deep Graph Library (DGL) object and constructs the data loaders (*e.g.*, neighbor samplers) for the node embedding model. Key functions are `load_graph()` to load the KG into a `dgl.DGLGraph` heterograph object, `partition_graph()` to partition the KG into train, validation, and test sets, and `create_dataloaders()` to construct the dataloaders.

- 🤖 `models.py`: This script contains the node embedding model. Key classes are `BilinearDecoder` and `HGT`, which define the bilinear decoder and the Heterogeneous Graph Transformer (HGT) model, respectively.

- 🪄 `pretrain.py`: This script contains the main function to pretrain the HGT model. It reads the hyperparameters from `hyperparameters.py`, loads the KG into a DGL object using functions from `dataloaders.py`, and trains the HGT model defined in `models.py`.

- 🛠️ `utils.py`: This file contains utility functions.

All the above functions can be evaluated interactively and modified in a Jupyter notebook. Please see `test_model.ipynb`.