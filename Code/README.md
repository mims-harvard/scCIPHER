# scCIPHER Code

This directory contains the code for the scCIPHER project. The code is organized as follows.

### NeuroKG Construction

All `.Rmd` files in the `NeuroKG` directory are used to construct the NeuroKG knowledge graph.

### CZI CELLxGENE scRNA-seq Analysis

Scripts in the `CELLxGENE` directory are used to analyze and visualize the single-cell RNA-sequencing data from the CZI CELLxGENE Census. Various data retrieval strategies are implemented, including full retrieval, grouping, and chunking. Downstream analysis is performed using the R package Seurat and the Python package scanpy.

### Model Pre-Training

Pre-training of the node embedding model is defined in the `scCIPHER` directory.

⏳ Note that the code for the node embedding model is still under development.