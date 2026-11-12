
======================================================
Binferno: Contrastive Barlow-Twins metagenomic binning
======================================================

.. start-badges

|testing badge| |docs badge|

.. |testing badge| image:: https://github.com/Andyargueasae/binferno/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/Andyargueasae/binferno/actions

.. |docs badge| image:: https://github.com/Andyargueasae/binferno/actions/workflows/docs.yml/badge.svg
    :target: https://Andyargueasae.github.io/binferno

.. end-badges

.. image:: https://raw.githubusercontent.com/Andyargueasae/binferno/main/docs/source/_static/images/binferno-banner.jpg
    :alt: Binferno logo


Binferno is a python-based metagenomic binner that handles metagenomic contigs for both prokaryotic genomes and eukaryotic plastid genomes.


Installation
============

Binferno is installable via pip:

.. code-block:: bash

    pip install binferno

Or, you can install it via conda using conda-forge and bioconda channels:

.. code-block:: bash

    mamba create -y -n binferno-010 \
        --override-channels --strict-channel-priority \
        -c conda-forge -c bioconda \
        'python=3.10' 'binferno=0.1.0'

    
To verify installation, run:

.. code-block:: bash

    binferno --help
    

Input Data
================

Like most other metagenomic binners, Binferno uses two types of features: sequence composition and coverage. The sequence composition is represented by tetranucleotide frequencies (TNF), while the coverage is represented by the average depth of coverage across the contig.
But we also provide options to improve the binning:

 - **Canonical 7-rank taxonomy**: Binferno can use the taxonomy information of contigs to improve the binning results. The taxonomy information can be obtained from tools like Kraken2 or Kaiju. The taxonomy information should be provided in a tab-delimited file with two columns: contigs and taxonomy predictions.
 - **ESM-C protein language model**: Binferno also incorporates the ESM-C protein language model to extract features from the ORFs predicted on contigs. Through mean or attention pooling, the ESM-C features per contig can be involved in training and finally contribute to improved embeddings used for clustering.  

Usage
=====

You can check the versions of Binferno:

.. code-block:: bash

    binferno --version


The simplest command to run Binferno is:

.. code-block:: bash

    binferno --contigs $contig_file --bam-dir $bam_dir_path --output-dir $output_dir \
        --threads $num_threads --postprocessor.checkm-database $CHECKM_DATABASE \
        --training.use-simclr --features.coverage.comebin

Such command will run Binferno with most default parameters: encode contigs by canonical 136-dimension kmer features and coverage mean and standard deviation.
Training takes the better-performing SimCLR contrastive learning method, and the clustering will take Leiden and postprocessing module writes the MAGs to the output directory.

The output bins will be stored in the output directory specified by ``--output-dir``. The final bins will be stored in the subdirectory ``res_bins``.

Computing using GPU and CPU
===========================

Binferno will automatically detect if GPU is available and use it for training, we recommend using GPU as it will speed up the training process significantly. 

For more details, please refer to the documentation: https://andyargueasae.github.io/binferno/Binferno.html.

Credits
========

.. start-credits

Binferno is developed by:

- `PhD candidate Yuhao Tong <https://www.linkedin.com/in/yuhao-tong-2a015825b/>`_
- `A/Prof Heroen Verbruggen <https://hverbruggen.codeberg.page>`_
- `Dr Vanessa R. Marcelino <https://hologenomics.org/people.html>`_
- `Dr Robert Turnbull <https://robturnbull.com>`_

.. end-credits

Contact:
    - Yuhao Tong:
        - Email: tyuhao037@gmail.com
