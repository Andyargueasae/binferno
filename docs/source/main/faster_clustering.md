# Faster Leiden clustering

The RBER sweep uses seven explicit configurations instead of the previous
120-configuration Cartesian product:

| Bandwidth | Neighbours | Resolution | Pruning percentile |
| --- | --- | --- | --- |
| 0.1 | 15 | 1 | 100 |
| 0.1 | 15 | 5 | 100 |
| 0.1 | 15 | 1 | 80 |
| 0.2 | 15 | 1 | 100 |
| 0.05 | 15 | 0.5 | 100 |
| 0.05 | 100 | 0.5 | 50 |
| 0.05 | 100 | 1 | 50 |

Configurations with the same neighbour count, bandwidth, and pruning percentile
share one weighted igraph within a worker. This requires five graph builds for
the seven RBER runs. Each resolution starts from the original membership with
a fresh partition and optimiser, the same random seed, and optimisation to
convergence. HNSW construction, edge pruning and weighting, marker seeds,
contig-length node sizes, taxonomy penalties, and seeded KMeans are preserved.
Only candidates selected for the current sweep enter `ClusterResult`, even if
the output directory also contains TSVs from an earlier full sweep.

## Selection evidence

The shortlist was selected from eight user-supplied `estimate_res_*.txt` tables,
ranking configurations by `num_905` first and `sum_cont5` second. Every shortlisted
configuration was evaluated in all eight runs. The best metric pair available
within the shortlist matches that of the full table in every run:

| Table | Full Leiden configurations | Full sweep (`num_905`, `sum_cont5`) | Shortlist |
| --- | --- | --- | --- |
| estimate_res_1.txt | 216 | (59, 265) | (59, 265) |
| estimate_res_2.txt | 216 | (45, 146) | (45, 146) |
| estimate_res_3.txt | 216 | (62, 275) | (62, 275) |
| estimate_res_4.txt | 216 | (14, 58) | (14, 58) |
| estimate_res_5.txt | 120 | (11, 51) | (11, 51) |
| estimate_res_6.txt | 120 | (12, 53) | (12, 53) |
| estimate_res_7.txt | 216 | (8, 30) | (8, 30) |
| estimate_res_8.txt | 216 | (58, 281) | (58, 281) |

`num_905` counts bins with completeness >90 and contamination <5.
`sum_cont5` adds the counts above completeness 50, 70, and 90 at contamination
<5, so its categories overlap. These tables informed the shortlist and are not
an independent validation set. Matching counts does not establish identical
MAG membership or guarantee performance on future embeddings.

The production postprocessor still selects by `sum`, then `sum_cont5`; its
selection policy is unchanged. CPM also retains its original 120-configuration
sweep because these tables provide evidence for RBER only. CPM benefits from
graph reuse across its five resolutions, reducing graph builds to 24.
