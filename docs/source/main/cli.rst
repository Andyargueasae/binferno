================================
Binferno command line interface
================================

Binferno is a easy-to-use command line tool that covers the steps from feature extraction to clustering results writing of a metagenomic binning tool. 
Its command line interface (CLI) is wrapped by pydantic and tyro, which is not restricting all arguments to one main script. So that we can make this tool highly modular and easily customize which modules to use in the pipeline.

The full command line is listed as follows:

.. code-block:: text

    ╭─ options ─────────────────────────────────────────────────────────────────────────────────────────────╮
    │ -h, --help                                                                                            │
    │       show this help message and exit                                                                 │
    │ --contigs CONTIGS                                                                                     │
    │       Path to the input contigs in fasta format. (required)                                           │
    │ --bam-dir BAM_DIR                                                                                     │
    │       Path to the input directory of BAM files, must be paired with contigs. Used for calculating     │
    │       coverage mean and std. (required)                                                               │
    │ --output-dir OUTPUT_DIR                                                                               │
    │       Path to the output directory to store all results, including intermediate results and final     │
    │       bins. (required)                                                                                │
    │ --threads THREADS                                                                                     │
    │       Number of threads to use for parallel processing. (default: 4)                                  │
    │ --train-only TRAIN_ONLY, --no-train-only TRAIN_ONLY                                                   │
    │       True: only run training, False: full process. (default: False)                                  │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ features options ────────────────────────────────────────────────────────────────────────────────────╮
    │ --features.use-kmer, --features.no-use-kmer                                                           │
    │       Whether to use k-mer features for training. (default: True)                                     │
    │ --features.use-coverage, --features.no-use-coverage                                                   │
    │       Whether to use coverage features for training. (default: True)                                  │
    │ --features.use-taxonomy, --features.no-use-taxonomy                                                   │
    │       Whether to use taxonomy features for training. (default: False)                                 │
    │ --features.save-features, --features.no-save-features                                                 │
    │       Whether to save the generated features as tsv files in the output directory. (default: True)    │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ features.kmer options ───────────────────────────────────────────────────────────────────────────────╮
    │ --features.kmer.pseudocount INT                                                                       │
    │       Pseudocount to add to k-mer counts before normalization. Default is 1. (default: 1)             │
    │ --features.kmer.kmer-len INT                                                                          │
    │       Length of k-mers to compute. Default is 4. (default: 4)                                         │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ features.coverage options ───────────────────────────────────────────────────────────────────────────╮
    │ --features.coverage.rpkm-normalize, --features.coverage.no-rpkm-normalize                             │
    │       Whether to apply RPKM normalization for non-SimCLR paths. (default: True)                       │
    │ --features.coverage.comebin, --features.coverage.no-comebin                                           │
    │       Whether to use COMEBin-like normalization for SimCLR. (default: False)                          │
    │ --features.coverage.max-norm-stacked, --features.coverage.no-max-norm-stacked                         │
    │       Whether to apply max normalization for stacked coverage. (default: False)                       │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ features.taxonomy options ───────────────────────────────────────────────────────────────────────────╮
    │ Placeholder for future taxonomy feature builder                                                       │
    │ ───────────────────────────────────────────────────────────────────────────────────────────────────── │
    │ --features.taxonomy.taxonomy PATH                                                                     │
    │       Path to a TSV file containing taxonomic classifications for each contig. Columns: contig_id,    │
    │       taxonomy_string. (default: .)                                                                   │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ training options ────────────────────────────────────────────────────────────────────────────────────╮
    │ --training.use-simclr, --training.no-use-simclr                                                       │
    │       Whether to use SimCLR for training. If False, only Barlow Twins will be used for training.      │
    │       (default: False)                                                                                │
    │ --training.use-barlowtwins, --training.no-use-barlowtwins                                             │
    │       Whether to use Barlow Twins for training. If False, only SimCLR will be used for training.      │
    │       (default: False)                                                                                │
    │ --training.concat, --training.no-concat                                                               │
    │       Whether to concatenate the embeddings from SimCLR and Barlow Twins. If False, the embeddings    │
    │       are added and normalized. (default: True)                                                       │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ training.common options ─────────────────────────────────────────────────────────────────────────────╮
    │ --training.common.batch-size INT                                                                      │
    │       Training batch size. (default: 1024)                                                            │
    │ --training.common.epochs INT                                                                          │
    │       Number of training epochs. (default: 300)                                                       │
    │ --training.common.output-dim INT                                                                      │
    │       Embedding dimension. (default: 128)                                                             │
    │ --training.common.contig-length INT                                                                   │
    │       Minimum training contig length. (default: 1000)                                                 │
    │ --training.common.seed INT                                                                            │
    │       Random seed for training and augmentation. (default: 42)                                        │
    │ --training.common.get-individual-embs, --training.common.no-get-individual-embs                       │
    │       Save separate modality embeddings. (default: False)                                             │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ training.scg options ────────────────────────────────────────────────────────────────────────────────╮
    │ --training.scg.lambda-scg FLOAT                                                                       │
    │       SCG cannot-link loss weight; zero disables SCG detection. (default: 0.0)                        │
    │ --training.scg.cosine-margin FLOAT                                                                    │
    │       (default: 0.4)                                                                                  │
    │ --training.scg.scg-batch-size INT                                                                     │
    │       (default: 128)                                                                                  │
    │ --training.scg.scg-update-freq INT                                                                    │
    │       (default: 2)                                                                                    │
    │ --training.scg.scg-loss-warmup INT                                                                    │
    │       (default: 10)                                                                                   │
    │ --training.scg.scg-marker-hmm PATH                                                                    │
    │       (default: /data/scratch/projects/punim1293/andy/binferno/binferno/auxiliary/marker.hmm)         │
    │ --training.scg.scg-max-evalue FLOAT                                                                   │
    │       (default: 1e-10)                                                                                │
    │ --training.scg.scg-min-hmm-coverage FLOAT                                                             │
    │       (default: 0.75)                                                                                 │
    │ --training.scg.scg-hard-pairs INT                                                                     │
    │       Maximum hard marker pairs per update (Barlow Twins only). (default: 1024)                       │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ training.esmc options ───────────────────────────────────────────────────────────────────────────────╮
    │ --training.esmc.enabled, --training.esmc.no-enabled                                                   │
    │       Include the ESM-C protein modality. (default: False)                                            │
    │ --training.esmc.model {esmc_300m,esmc_600m}                                                           │
    │       (default: esmc_300m)                                                                            │
    │ --training.esmc.pooling {mean,attention}                                                              │
    │       (default: mean)                                                                                 │
    │ --training.esmc.fusion {residual,concat}                                                              │
    │       Late protein residual or concatenation before a joint MLP. (default: residual)                  │
    │ --training.esmc.augmentation {shared,coordinate}                                                      │
    │       Pool complete ORFs in each fragment; shared is the baseline. (default: coordinate)              │
    │ --training.esmc.num-heads INT                                                                         │
    │       (default: 4)                                                                                    │
    │ --training.esmc.weight FLOAT                                                                          │
    │       Protein residual weight. (default: 0.2)                                                         │
    │ --training.esmc.batch-size INT                                                                        │
    │       Protein chunks per ESM-C batch. (default: 4)                                                    │
    │ --training.esmc.max-residues INT                                                                      │
    │       (default: 1022)                                                                                 │
    │ --training.esmc.device STR                                                                            │
    │       ESM-C inference device; auto, cpu or cuda. (default: auto)                                      │
    │ --training.esmc.pool-max-padded-tokens INT                                                            │
    │       Max padded tokens for attention pooling; larger values use more GPU memory. (default: 65536)    │
    │ --training.esmc.pool-max-proteins INT                                                                 │
    │       Max proteins per attention pooling batch; larger values use more GPU memory. (default: 8192)    │
    │ --training.esmc.pool-max-views INT                                                                    │
    │       Max contig/view pairs per attention pooling batch; larger values use more GPU memory. (default: │
    │       512)                                                                                            │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ training.simclr options ─────────────────────────────────────────────────────────────────────────────╮
    │ --training.simclr.simclrnet, --training.simclr.no-simclrnet                                           │
    │       Whether to use SimCLRNet architecture for training. If False, use a simpler MLP architecture.   │
    │       (default: True)                                                                                 │
    │ --training.simclr.proj, --training.simclr.no-proj                                                     │
    │       Whether to use a projection head in the NeuralNetwork architecture. (default: False)            │
    │ --training.simclr.learning-rate FLOAT                                                                 │
    │       Learning rate for training. (default: 0.001)                                                    │
    │ --training.simclr.view-num INT                                                                        │
    │       Number of augmented views to generate for each contig. (default: 6)                             │
    │ --training.simclr.attention, --training.simclr.no-attention                                           │
    │       Whether to use attention fusion in the Fusion module instead of gated fusion. (default: False)  │
    │ --training.simclr.temperature {None}|FLOAT                                                            │
    │       Temperature parameter for the InfoNCE loss. By default, it is set based on the N50 of the input │
    │       contigs. If the N50 is greater than 10,000, the temperature is set to 0.07; otherwise, it is    │
    │       set to 0.15. (default: None)                                                                    │
    │ --training.simclr.max-lr-epochs INT                                                                   │
    │       Number of epochs to wait before resuming learning rate scheduling after reaching the maximum    │
    │       learning rate. (default: 10)                                                                    │
    │ --training.simclr.hard-weighted-nce, --training.simclr.no-hard-weighted-nce                           │
    │       Whether to use the hard weighted InfoNCE loss instead of the standard InfoNCE loss. This loss   │
    │       gives more weight to hard negative samples during training. (default: False)                    │
    │ --training.simclr.get-covout, --training.simclr.no-get-covout                                         │
    │       Whether to save coverage embeddings separately. (default: False)                                │
    │ --training.simclr.auxiliary-loss, --training.simclr.no-auxiliary-loss                                 │
    │       Whether to include an auxiliary loss on the individual view embeddings in addition to the main  │
    │       InfoNCE loss on the fused embeddings. This can help encourage the model to learn useful         │
    │       representations in each view before fusion. (default: False)                                    │
    │ --training.simclr.alpha-tax FLOAT                                                                     │
    │       The weight applied to taxonomic embeddings within the neural network. (default: 0.1)            │
    │ --training.simclr.get-taxout, --training.simclr.no-get-taxout                                         │
    │       The switch for getting taxonomy embeddings. (default: False)                                    │
    │ --training.simclr.bin-tax-emb, --training.simclr.no-bin-tax-emb                                       │
    │       Decide whether to use taxonomy features only to get the resulting bins. (default: False)        │
    │ --training.simclr.hiddim INT                                                                          │
    │       The dimension of the hidden layers in the SimCLRNet architecture. This is used to project the   │
    │       embeddings into a lower-dimensional space for contrastive learning. (default: 2048)             │
    │ --training.simclr.dropout FLOAT                                                                       │
    │       Dropout rate for the neural network layers. This is used to prevent overfitting during          │
    │       training. (default: 0.2)                                                                        │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ training.barlowtwins options ────────────────────────────────────────────────────────────────────────╮
    │ --training.barlowtwins.learning-rate FLOAT                                                            │
    │       Learning rate for training. (default: 0.0025)                                                   │
    │ --training.barlowtwins.view-num INT                                                                   │
    │       Number of augmented views to generate for each contig. (default: 8)                             │
    │ --training.barlowtwins.warmup-epochs INT                                                              │
    │       Number of epochs to linearly increase the learning rate from a small value to the initial       │
    │       learning rate at the start of training. (default: 10)                                           │
    │ --training.barlowtwins.max-lr-epochs INT                                                              │
    │       Number of epochs to wait before resuming learning rate scheduling after reaching the maximum    │
    │       learning rate. (default: 0)                                                                     │
    │ --training.barlowtwins.proj-head-dim INT                                                              │
    │       Dimension of the projection head used in the Barlow Twins architecture. The projection head is  │
    │       a small neural network that maps the output of the main encoder to a space where the Barlow     │
    │       Twins loss is applied. A larger proj_head_dim can allow for more complex transformations and    │
    │       potentially better performance, but it also increases the number of parameters and              │
    │       computational cost of the model. (default: 2048)                                                │
    │ --training.barlowtwins.weight-tau FLOAT                                                               │
    │       Length-precision exponent for the Barlow Twins loss. 0 reproduces the unweighted loss exactly.  │
    │       0.5 is the recommended operating point; above 0.75 the effective batch size collapses.          │
    │       (default: 0.0)                                                                                  │
    │ --training.barlowtwins.weight-clip-percentile FLOAT                                                   │
    │       Percentile cap on raw pair precision before tempering. (default: 99.0)                          │
    │ --training.barlowtwins.weight-per-contig-norm, --training.barlowtwins.no-weight-per-contig-norm       │
    │       Renormalise weights so each contig's pairs keep their unweighted total, so tau acts within      │
    │       contigs only. (default: True)                                                                   │
    │ --training.barlowtwins.stratified-cap, --training.barlowtwins.no-stratified-cap                       │
    │       Guarantee every contig keeps one positive pair when the pair set exceeds max_positive_pairs.    │
    │       False reproduces the uniform sampler, which starved 7,423 of 131,042 contigs on a real run.     │
    │       (default: True)                                                                                 │
    │ --training.barlowtwins.alpha-tax FLOAT                                                                │
    │       A small weight used to constrain taxonomy embedding fusion. (default: 0.1)                      │
    │ --training.barlowtwins.lambda-taxonomy FLOAT                                                          │
    │       Weight of the masked multi-rank taxonomy prediction loss. Zero disables the auxiliary loss      │
    │       while retaining taxonomy embedding fusion. (default: 0.1)                                       │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ clustering options ──────────────────────────────────────────────────────────────────────────────────╮
    │ --clustering.plastid, --clustering.no-plastid                                                         │
    │       Whether to use plastid-specific seed generation and minimum bin size. (default: False)          │
    │ --clustering.longread, --clustering.no-longread                                                       │
    │       Whether the input data is from long-read sequencing. This will impact the downstream binning    │
    │       choices of parameters and thresholds. (default: False)                                          │
    │ --clustering.cluster-method {leiden,dbscan}                                                           │
    │       The clustering method to use. DBSCAN is DEPRECATED and will be removed in future versions. Use  │
    │       Leiden instead. (default: leiden)                                                               │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ clustering.leiden options ───────────────────────────────────────────────────────────────────────────╮
    │ --clustering.leiden.plastid, --clustering.leiden.no-plastid                                           │
    │       Whether to use plastid-specific seed generation and minimum bin size. (default: False)          │
    │ --clustering.leiden.longread, --clustering.leiden.no-longread                                         │
    │       Whether the input data is from long-read sequencing. This will impact the downstream binning    │
    │       choices of parameters and thresholds. (default: False)                                          │
    │ --clustering.leiden.seed INT                                                                          │
    │       (default: 42)                                                                                   │
    │ --clustering.leiden.mutual, --clustering.leiden.no-mutual                                             │
    │       Whether to use mutual kNN filtering in Leiden clustering. (default: False)                      │
    │ --clustering.leiden.contig-len INT                                                                    │
    │       Minimum contig length for seed generation. (default: 1000)                                      │
    │ --clustering.leiden.cluster-min-length INT                                                            │
    │       Minimum contig length for phase-1 clustering. (default: 1000)                                   │
    │ --clustering.leiden.CPM, --clustering.leiden.no-CPM                                                   │
    │       Whether to use CPM (Constant Potts Model) for Leiden clustering instead of RBER                 │
    │       (Reichardt-Bornholdt) [DEPRECATED]. (default: False)                                            │
    │ --clustering.leiden.taxonomy-edge-factor FLOAT                                                        │
    │       Genus conflict edge multiplier; 1.0 disables taxonomy refinement. (default: 1.0)                │
    │ --clustering.leiden.max-workers INT                                                                   │
    │       Maximum concurrent Leiden runs; 0 selects from CPU and RAM budgets. (default: 0)                │
    │ --clustering.leiden.worker-memory-gb FLOAT                                                            │
    │       Estimated peak additional GiB per Leiden worker; 0 estimates from graph size. (default: 0.0)    │
    │ --clustering.leiden.memory-reserve-fraction FLOAT                                                     │
    │       Fraction of remaining RAM reserved when selecting Leiden workers. (default: 0.25)               │
    │ --clustering.leiden.sweep-config STR                                                                  │
    │       Sweep configuration for Leiden clustering; 'default' or 'mini'. (default: default)              │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ clustering.dbscan options ───────────────────────────────────────────────────────────────────────────╮
    │ --clustering.dbscan.plastid, --clustering.dbscan.no-plastid                                           │
    │       Whether to use plastid-specific seed generation and minimum bin size. (default: False)          │
    │ --clustering.dbscan.longread, --clustering.dbscan.no-longread                                         │
    │       Whether the input data is from long-read sequencing. This will impact the downstream binning    │
    │       choices of parameters and thresholds. (default: False)                                          │
    │ --clustering.dbscan.hdbscan, --clustering.dbscan.no-hdbscan                                           │
    │       Use HDBSCAN instead of DBSCAN. [DEPRECATED: HDBSCAN is not recommended for binning] (default:   │
    │       False)                                                                                          │
    │ --clustering.dbscan.core-bp {None}|INT                                                                │
    │       Assembled bp required within eps for a contig to be a core point. Portable across assembly      │
    │       sizes; set None to fall back to classic contig-count min_samples. (default: 100000)             │
    │ --clustering.dbscan.min-bin-bp INT                                                                    │
    │       Discard candidate bins smaller than this. (default: 100000)                                     │
    │ --clustering.dbscan.contig-len INT                                                                    │
    │       Contigs shorter than this are ignored for clustering. (default: 1000)                           │
    │ --clustering.dbscan.marker-hmm {None}|PATH                                                            │
    │       Path to the .hmm file defining the marker vocabulary. Strongly recommended: without it,         │
    │       completeness is normalised by the markers that happened to hit this assembly and is not         │
    │       comparable between runs. (default:                                                              │
    │       /data/scratch/projects/punim1293/andy/binferno/binferno/auxiliary/bacar_marker.hmm)             │
    │ --clustering.dbscan.archaeal-hmm {None}|PATH                                                          │
    │       Optional second marker set; completeness takes the best domain. (default: None)                 │
    │ --clustering.dbscan.max-marker-evalue FLOAT                                                           │
    │       (default: 1e-05)                                                                                │
    │ --clustering.dbscan.recruit, --clustering.dbscan.no-recruit                                           │
    │       Recruit leftover contigs into nearby bins. (default: True)                                      │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ postprocessor options ───────────────────────────────────────────────────────────────────────────────╮
    │ --postprocessor.checkm-database PATH|STR                                                              │
    │       Path to CheckM database (default: '')                                                           │
    │ --postprocessor.reuse-checkm-profile, --postprocessor.no-reuse-checkm-profile                         │
    │       Reuse matching, successfully completed CheckM profiling steps. (default: True)                  │
    │ --postprocessor.write-checkm-quality-reports, --postprocessor.no-write-checkm-quality-reports         │
    │       Write CheckM QA summary reports; disable to run only marker QA needed for selection. (default:  │
    │       True)                                                                                           │
    │ --postprocessor.minbin-size INT                                                                       │
    │       Minimum bin size for filtering contigs before clustering. (default: 200000)                     │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ reducer options ─────────────────────────────────────────────────────────────────────────────────────╮
    │ --reducer.use-pca, --reducer.no-use-pca                                                               │
    │       Whether to apply PCA for dimension reduction. (default: False)                                  │
    │ --reducer.use-umap, --reducer.no-use-umap                                                             │
    │       Whether to apply UMAP for dimension reduction. (default: False)                                 │
    │ --reducer.use-pipeline, --reducer.no-use-pipeline                                                     │
    │       Whether to apply a pipeline of PCA and UMAP for dimension reduction. (default: False)           │
    │ --reducer.default-dim INT                                                                             │
    │       Default dimension for dimension reduction. (default: 32)                                        │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ reducer.umap options ────────────────────────────────────────────────────────────────────────────────╮
    │ --reducer.umap.n-neighbors INT                                                                        │
    │       The number of neighbors parameter for UMAP. (default: 30)                                       │
    │ --reducer.umap.min-dist FLOAT                                                                         │
    │       The minimum distance parameter for UMAP. (default: 0.1)                                         │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯


We have five modules: feature, training, clustering, postprocessor and reducer. Each module has its own set of options, which can be accessed by running ``binferno --help``.