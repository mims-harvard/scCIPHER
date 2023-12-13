<!-- scCIPHER: Contextual Deep Learning on Single-Cell-Enriched Knowledge Graphs for Precision Medicine in Neurological Disorders -->
<p align="center">
<img src="Results/scCIPHER_header.png?raw=true" width="100%" title="scCIPHER: Contextual Deep Learning on Single-Cell-Enriched Knowledge Graphs for Precision Medicine in Neurological Disorders">
</p>

## Project Summary

Neurological disorders are the leading driver of global disability and cause 16.8% of global mortality. Unfortunately, most lack disease-modifying treatments or cures. To address disease complexity and heterogeneity in neurological disease, we developed scCIPHER, an AI approach for Contextually Informed Precision HEalthcaRe using deep learning on single-cell-enriched knowledge graphs. First, we constructed the Neurological Disease Knowledge Graph (NeuroKG), a neurobiological knowledge graph with 132K nodes and 3.98 million edges, by integrating 20 high-quality primary data sources with single-cell RNA-sequencing data from 3.37 million cells across 106 regions of the adult human brain. Next, we pre-trained a heterogeneous graph transformer on NeuroKG to create scCIPHER. We leverage scCIPHER to make precision medicine-based predictions in neurological disorders across patient phenotyping, therapeutic response prediction, and causal gene discovery tasks, with validation in large-scale patient cohorts.


<p align="center">
<img src="Results/CELLxGENE/non_neuron_markers.png?raw=true" width="80%" title="Analysis of 888,263 non-neuronal cells (e.g., astrocytes, microglia, oligodendrocytes, oligodendrocyte precursor cells, ependymal cells, and vascular cells).">
</p>

*Single-cell RNA-sequencing data from Siletti* et al. *in* Science, *2023 (DOI: [10.1126/science.add7046](https://doi.org/10.1126/science.add7046))*

## Dependencies

To run the code, please install:
* The [Python](https://www.python.org/) programming language.
* The [R](https://www.r-project.org/) programming language and statistical computing environment (as well as the [RStudio](https://rstudio.com/) integrated development environment).

Individual dependencies are also specified in each script. Along with data manipulation and visualization packages, these include:
* The [PyTorch](https://pytorch.org/) open source machine learning framework for Python.
* The [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) library for geometric deep learning on graphs and manifolds.
* The [PyTorch Lightning](https://www.pytorchlightning.ai/) lightweight PyTorch wrapper for high-performance AI research.

Activate the `scCIPHER_env` virtual environment with the following:

```
source setup.sh
```
If desired, a Jupyter kernel can be created with the following:

```
source setup_jupyter.sh
```
