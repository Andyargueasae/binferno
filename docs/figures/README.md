# Binferno workflow

Landscape 1120 × 630 SVG and editable draw.io source, created using the draw.io MCP.

Based on the current working tree inspected 2026-10-20: binferno/main.py, features/__init__.py, training/{__init__,barlowtwins,simclr}.py, reducer/__init__.py, clustering/leiden.py and postprocessing/{__init__,writer_utils}.py.

The figure simplifies the main Leiden pipeline. Clustering includes marker-seeded k-means and a sweep of Leiden candidates. Quality assessment selects the best grouping and filters undersized bins. Taxonomy remains separate from continuous features; Barlow Twins permits taxonomy guidance at training or Leiden clustering, not both. Marker guidance during learning and PCA/UMAP are optional. Alternative encoders and deprecated DBSCAN are omitted for clarity. Genome bins are collections of contigs, not guaranteed complete genomes.

The SVG embeds draw.io XML for reopening in diagrams.net.

