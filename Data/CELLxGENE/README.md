# Single-Cell RNA-sequencing Data

I include data from the following papers:
1. "Transcriptomic diversity of cell types across the adult human brain" by Siletti *et al.* (senior author, Sten Linnarsson) published in *Science*, 2023 (DOI: [10.1126/science.add7046](https://doi.org/10.1126/science.add7046)).
2. "Integrative single-cell analysis of transcriptional and epigenetic states in the human adult brain" by Lake *et al.*, published in *Nature Biotechnology*, 2017 (DOI: [10.1038/nbt.4038](https://www.nature.com/articles/nbt.4038)).
3. "Purification and Characterization of Progenitor and Mature Human Astrocytes Reveals Transcriptional and Functional Differences with Mouse" by Zhang *et al.*, published in *Neuroscience*, 2016 (DOI: [10.1016/j.neuron.2015.11.013](https://doi.org/10.1016/j.neuron.2015.11.013)). See also [Brain RNA-seq.org](http://www.brainrnaseq.org/).

## Linnarsson *et al.*

The Linnarsson *et al.* dataset has:
* 31 superclusters, 461 clusters, and 3313 subclusters already identified.
* 106 anatomically distinct dissections across ten brain regions.
I can create a new node for each cluster, subcluster, and dissection region, as well as nodes for the brain regions if they do not already exist in UBERON. 

To download the dataset, run the following commands in the `Data/cell_type` directory:

```bash
curl https://storage.googleapis.com/linnarsson-lab-human/Neurons.h5ad -o linnarsson/Neurons.h5ad
curl https://storage.googleapis.com/linnarsson-lab-human/Nonneurons.h5ad -o linnarsson/Nonneurons.h5ad
```

Data can also be retrieved from the [CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/) from the Chan Zuckeberg Initiative.


## Lake *et al.*

I analyze Lake *et al.* using differential expression results from all clusters reported in Table S3. From the paper:
> We used Seurat software (V1.4.0.5) in R (https://github.com/satijalab/seurat) to construct violin plots and carry out differential gene expression analyses. For normalization, UMI counts for all annotated nuclei were scaled by the total UMI counts (excluding mitochondrial genes), multiplied by 10,000, and transformed to log space. Technical effects associated with UMI coverage and batch identity were regressed from scaled data with the RegressOut function in Seurat. Genes that were differentially expressed between cell types and subtypes were identified (Seurat software) by a likelihood-ratio test on all genes to identify 0.25-fold (log scale) enriched genes detected in at least 25% of cells in the cluster. Differential expression analyses were performed for all clusters, for excitatory or inhibitory neuron subtypes separately, for cerebellar data sets separately, or for all oligodendrocyte lineage cells separately (Supplementary Table 3).

See also Allen Brain Atlas data at:
* [All regions](https://portal.brain-map.org/atlases-and-data/rnaseq)
* [SMART-seq on multiple cortical areas](https://portal.brain-map.org/atlases-and-data/rnaseq/human-multiple-cortical-areas-smart-seq)
