============================
Running Binferno/运行Binferno
============================

Vanilla Runs: Binferno with ESM-C
=================================

With pooling method mean:

.. code-block:: bash

    binferno --contigs $contig_file --bam-dir $bam_dir_path --output-dir $output_dir \
    --threads $num_threads --postprocessor.checkm-database $CHECKM_DATABASE \
    --training.use-simclr --features.coverage.comebin \
    --training.esmc.enabled --training.esmc.pooling "mean" \
    --training.common.epochs 200

With pooling method "attention":

.. code-block:: bash
    
    binferno --contigs $contig_file --bam-dir $bam_dir_path --output-dir $output_dir \
        --threads $num_threads --postprocessor.checkm-database $CHECKM_DATABASE \
        --training.use-simclr --features.coverage.comebin \
        --training.esmc.enabled --training.esmc.pooling "attention" \
        --training.common.epochs 200


Run Binferno with taxonomy
==========================

SimCLR:

.. code-block:: bash

    binferno --contigs "$contig_file" --bam-dir "$bam_dir_path" --output-dir "$output_dir" \
    --threads "$num_threads" --postprocessor.checkm-database "$CHECKM_DATABASE" \
    --training.use-simclr --features.coverage.comebin \
    --training.simclr.learning-rate 1e-3 --training.simclr.max-lr-epochs 10 \
    --features.use-taxonomy --features.taxonomy.taxonomy "$taxonomy" \
    --training.common.epochs 200


Barlow Twins: we don't recommend running taxonomy features with barlow-twins, as the internal processing of features cannot handle the discriminative taxonomic features and canonical features well. We will seek chances to find new solutions and update the branch asap.  

If you want to use barlow-twins branch
=======================================

.. code-block:: bash

    binferno --contigs $contig_file --bam-dir $bam_dir_path --output-dir $output_dir --threads $num_threads \
        --postprocessor.checkm-database $CHECKM_DATABASE \
        --training.barlowtwins.learning-rate 1e-3 \
        --training.use-barlowtwins --training.barlowtwins.warmup-epochs 10 \
        --training.barlowtwins.proj-head-dim 2048 \
        --reducer.use-pipeline --reducer.default-dim 10

If you want to use reducer to reduce dimensions of output embeddings
====================================================================

.. code-block:: bash

    binferno --contigs $contig_file --bam-dir $bam_dir_path --output-dir $output_dir --threads $num_threads \
        --postprocessor.checkm-database $CHECKM_DATABASE \
        ... \
        --reducer.use-pipeline --reducer.default-dim 10

If you want to minimize the sweep size of clustering
====================================================

.. code-block:: bash

    binferno --contigs $contig_file --bam-dir $bam_dir_path --output-dir $output_dir \
        --threads $num_threads ... --clustering.leiden.sweep-config "mini"

Performance
===========
CAMI2 marine long-read GSA sample:

TBC...