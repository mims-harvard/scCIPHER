# NeuroKG Construction

This directory contains files output by NeuroKG construction scripts. Specifically:

1. `1_filter_KG`: Remove non-neuronal anatomy nodes from the KG by intersecting with the Uberon `nervous-minimal` dataset.

2. `2_enrich_KG`: Enrich KG by constructing cell-type nodes and edges from brain snRNA-seq data.

3. `3_harmonize_KG`: Harmonize identifiers and save the final knowledge graph.

4. `4_visualize_KG`: Visualize the knowledge graph and compute KG statistics.

Note that `.RDS` files are excluded from version control.