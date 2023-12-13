#!/bin/bash
module load gcc/9.2.0 cuda/11.7 python/3.9.14 git/2.9.5
# module load gcc/9.2.0 python/3.9.14 R/4.2.1 geos/3.10.2 git/2.9.5  # for Seurat
conda deactivate
source scCIPHER_env/bin/activate
jupyter notebook --port=54321 --browser='none'
