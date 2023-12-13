# Human Protein Atlas

The Human Protein Atlas (HPA) Brain Atlas, described in "An atlas of the protein-coding genes in the human, pig, and mouse brain" published in *Science*, 2020 (DOI: 10.1126/science.aay5947)[https://www.science.org/doi/10.1126/science.aay5947]), was used to define genes expressed in the brain.

## Genes Expressed in Brain

From the HPA website:

Protein-coding genes are classified based on RNA expression in brain from two different perspectives:
1. A whole-body perspective, comparing gene expression in the brain to peripheral organ and tissue types
2. A brain-centric point of view comparing gene expression in the various regions of the brain

Brain expression is compared to other organs and tissues by using the highest expression value of all brain regions. For the regional classification the brain is divided into 13 anatomically defined regions, color coded in Figure 1. The transcriptome analysis shows that 82% (n=16465) of all human proteins (n=20090) are expressed in the brain (based on 13 brain regions, spinal cord and corpus callosum). Regional classification was based on 16465 genes are detected in the brain and included in all used external datasets. Out of the genes with regional expression classification, 1055 are categorized as genes with a regionally elevated expression.

## HPA Queries
1. Expressed in brain (n=16465): [NOT tissue_category_rna:brain;not detected](https://www.proteinatlas.org/search/NOT+tissue_category_rna:brain;not+detected). This search query is saved as `brain_genes.tsv`.
2. Elevated expression in specific brain regions (n=1055): [brain_category_rna:Any;Region enriched,Group enriched,Region enhanced](https://www.proteinatlas.org/search/brain_category_rna%3AAny%3BRegion+enriched%2CGroup+enriched%2CRegion+enhanced) This search query is saved as `brain_region_genes.tsv`.


