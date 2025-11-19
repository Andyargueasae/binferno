#!/bin/bash

##########################################################################################################
###### This script is meant to be run COMEBin after obtaining the bam files with Corgi-generated augmentated data.

output_dir="/home/yuhtong/scratch/andy/CAMI_datasets/Corgi_COMEBin_gsa_marine_s0"
contig_file="/home/yuhtong/scratch/andy/simulation_short_read/2018.08.15_09.49.32_sample_0/contigs/anonymous_gsa.fasta"
bam_file_path="/home/yuhtong/scratch/andy/simulation_short_read/2018.08.15_09.49.32_sample_0/bam" # dummy.
temperature=0.15
emb_szs_forcov=2048
emb_szs=2048
batch_size=1024
n_views=6
num_threads=10

# cd Corgi_MLP_CL/COMEBin

# python main.py generate_aug_data --contig_file ${contig_file} \
#     --out_augdata_path ${output_dir}/data_augmentation \
#     --n_views ${n_views} --bam_file_path ${bam_file_path} --num_threads ${num_threads}


# CHECKPOINT="/data/gpfs/projects/punim1293/rob/corgi-order-large.ckpt"
# module load GCCcore/11.3.0; module load Python/3.10.4;
# source /home/yuhtong/scratch/andy/binferno/.venv/bin/activate
CHECKPOINT_BASE=/data/gpfs/projects/punim1293/rob/corgi-order-base.ckpt
# CHECKPOINT_NEW_LARGE=/data/scratch/projects/punim1293/andy/corgi-July30-plastidplus.ckpt
# cd /home/yuhtong/scratch/andy/maximum_metag_deepseq_sc97/data_augmentation/

# for aug in $(ls -d aug*/); do
#     corgi --input $aug/sequences_$(basename $aug).fasta --checkpoint $CHECKPOINT_NEW_LARGE \
#         --embeddings $aug/corgi_plastid_plus_$(basename $aug).npy \
#         --min-length 200 --output-csv $aug/corgi_plastid_plus_$(basename $aug).csv
# done

# deactivate
# module purge;

# module load Anaconda3/2024.02-1;
# # activate the conda environment
# eval "$(conda shell.bash hook)"
# # set the path to the conda environments of your project.
# export CONDA_ENVS_PATH="/home/yuhtong/scratch/andy/cd
# export CONDA_PKGS_DIR="/home/yuhtong/scratch/andy/cd/pkgs"

# conda activate comebin_env

# ##################################################################################################
# ###### Sync all views' contig data.
# ##################################################################################################

# # Run COMEBin training with Corgi's generated data. 
# python main.py train --data ${output_dir}/data_augmentation \
#     --temperature ${temperature} --emb_szs_forcov ${emb_szs_forcov} \
#     --batch_size ${batch_size} --emb_szs ${emb_szs} --n_views ${n_views} \
#     --add_model_for_coverage \
#     --output_path ${output_dir}/comebin_res_corgi --earlystop --addvars --vars_sqrt --num_threads ${num_threads}

# ########################################################################################################
# ###### Binning with COMEBin representations
# ########################################################################################################
# emb_file=${output_dir}/comebin_res_corgi/embeddings.tsv
# seed_file=${contig_file}.pltd_marker.2quarter_lencutoff_1001.seed

# python main.py bin --contig_file ${contig_file} \
#     --emb_file ${emb_file} \
#     --output_path ${output_dir}/comebin_res_corgi \
#     --seed_file ${seed_file} --num_threads ${num_threads}

########################################################################################################
###### Bac-arc version
########################################################################################################
emb_file=${output_dir}/comebin_res/embeddings.tsv
seed_file=${contig_file}.bacar_marker.2quarter_lencutoff_1001.seed

python main.py bin --contig_file ${contig_file} \
--emb_file ${emb_file} \
--output_path ${output_dir}/comebin_res \
--seed_file ${seed_file} --num_threads ${num_threads}

python main.py get_result --contig_file ${contig_file} \
--output_path ${output_dir}/comebin_res \
--seed_file ${seed_file} --num_threads ${num_threads} \
--checkm_database ${checkm_database}

if [[ $? -ne 0 ]] ; then echo "Something went wrong with running clustering. Exiting.";exit 1; fi


# #########################################################################################################
# ###### Get final binning result with COMEBin
# #########################################################################################################
# python main.py get_result --contig_file ${contig_file} \
#     --output_path ${output_dir}/comebin_res \
#     --seed_file ${seed_file} --num_threads ${num_threads} \
#     --checkm_database ${checkm_database}

# if [[ $? -ne 0 ]] ; then echo "Something went wrong with running clustering. Exiting.";exit 1; fi
