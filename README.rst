
=====================================================
Binferno: Contrastive Barlow-Twins metagenomic binning
=====================================================

.. start-badges

|testing badge| |docs badge|

.. |testing badge| image:: https://github.com/Andyargueasae/chloroscan/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/Andyargueasae/chloroscan/actions

.. |docs badge| image:: https://github.com/Andyargueasae/chloroscan/actions/workflows/docs.yml/badge.svg
    :target: https://Andyargueasae.github.io/chloroscan
    
.. end-badges



Binferno is a python-based metagenomic binner that handles metagenomic contigs for both prokaryotic genomes and eukaryotic plastid genomes. 

Installation
============

Binferno is installable via pip. But before installing it, build the conda environment using the conda_env.yaml provided.

.. code-block:: bash
    
    mamba env create -n Binferno_env -c conda-forge -c bioconda -c defaults python=3.10
    # mamba env create -n Binferno_env -f conda_env.yaml
    conda activate Binferno_env

Then, install Binferno via pip:

.. code-block:: bash

    pip install binferno

Check the installation by running:

.. code-block:: bash

    Binferno --help

Input features
================

Like most other metagenomic binners, Binferno uses two types of features: sequence composition and coverage. The sequence composition is represented by tetranucleotide frequencies (TNF), while the coverage is represented by the average depth of coverage across the contig.

For improved training, we also offer an option to use the projected kmer features: by multiplying original composition features with the kernel matrix obtained from VAMB.

Corgi encoding is also supported, which is a more compact representation of the sequence composition features with hidden taxonomic information encoded in the latent space.

Because running Binferno requires checkm to evaluate the quality of the bins, please specify the path to the checkm database in the command line when running Binferno.

For prokaryote binning: ``checkm_data.tar.gz`` from the official checkm repository can work. For plastid binning, we curated one database with plastid protein HMMs, which can be downloaded from figshare: ...

Usage
=====

A simple usage of Binferno is as follows:

.. code-block:: bash
    
    binferno --contig-file contigs.fasta --bam-dir path/to/bam/files \
        --output-dir path/to/output/dir --checkm-db path/to/checkm/db \
        --contig-length 1000

Such command will run Binferno with most default parameters: encode contigs by canonical 136-dimension kmer features and coverage mean and standard deviation.

For plastid binning, run with the flag ``--plastid``:

.. code-block:: bash
    
    binferno --contig-file contigs.fasta --bam-dir path/to/bam/files \
        --output-dir path/to/output/dir --checkm-db path/to/plastid/checkm/db \
        --contig-length 1000 --plastid

The output bins will be stored in the output directory specified by ``--output-dir``. The final bins will be stored in the subdirectory ``res_bins``.

Besides, other steps, like augmentations, training, and clustering outputs will also be stored as intermediate results in the output directory as ``augmentations``, ``training`` and ``cluster_res`` subdirectories, respectively.

For more details, please refer to the documentation: https://andyargueasae.github.io/chloroscan/Binferno.html.

Credits
=====

Binferno is developed by Yuhao Tong and Robert Turnbull. 

