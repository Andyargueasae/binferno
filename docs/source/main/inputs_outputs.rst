===========================
Inputs and Outputs/输入和输出
===========================

The raw inputs Binferno requires are like all other binners: contigs in FASTA format and reads mapping coverage profiles in sorted BAM format.

But using these features was shown to have an upper boundary of performance improvements, unless new modalities that represents the genomic discrimination can enter the analyses.

Hence, we introduce two new modalities we feed into Binferno:

 - Canonincal 7-rank taxonomy, from domain to species if availabe;

 - ESM-C encoded protein latents aligned with the sliced views;

The two new features both show improved binning outcomes in samples especially using long-read platforms.

Preparation
===========

For taxonomic features, we require a tsv file saving two columns: contig id and taxonomic predictions with headers: contigs and predictions.

This can be achieved by mmseqs2 easy-taxonomy:

.. code-block:: bash

    mmseqs easy-taxonomy $contig_file $DB_PATH $results $TMP_DIR --threads $THREADS --split-memory-limit $SPLIT_LIMIT_CPU \
    --tax-lineage 1 

To achieve, you will need to download a database that mmseqs2 generated, or find that database in your affiliated server (the size of the database is commonly large). Make sure to add ``--tax-lineage 1`` in the command to output the 7-rank taxonomy.

Once ready, prepare the taxonomy table using the following commands:

.. code-block:: bash

    echo -e "contigs\tpredictions" > taxonomy.tsv
    paste <(cut -f1 output_mmseqs_lca.tsv) <(cut -f9 output_mmseqs_lca.tsv) >> taxonomy.tsv

For ESM-C, good news, you don't need to do anything to prepare this. We will install esm package along with Binferno and it will detect the cli commands to begin encoding the predicted ORFs automatically. 