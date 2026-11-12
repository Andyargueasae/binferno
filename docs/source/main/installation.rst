================
Installation/安装
================

To install Binferno, you can use pip:

.. code-block:: bash

   pip install binferno

This will install the most updated binferno version, ensuring the best performance and runtime efficiency. 

Alternatively, you can also try conda to install it. 

.. code-block:: bash

    mamba create -y -n binferno-010 \
        --override-channels --strict-channel-priority \
        -c conda-forge -c bioconda \
        'python=3.10' 'binferno=0.1.0'

For now binferno is both up to date in both pypi releases and conda releases. We will notify if there are any changes in the future. 

Other data configurations
=========================

Binferno requires CheckM (checkm-genome in PyPI) in the postprocessor module and quality-analyze the MAGs. Make sure you download the released CheckM package. See: https://ecogenomics.github.io/CheckM/installing/reference_data.html.