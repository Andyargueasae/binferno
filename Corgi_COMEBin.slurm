#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=Corgi_Comebin

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 5:00:00

#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --account=punim1293

# Specify your email address to be notified of progress.
#SBATCH --mail-user=yuhao.tong@student.unimelb.edu.au
#SBATCH --mail-type=ALL

module purge;
module load Anaconda3/2024.02-1;

# activate the conda environment
eval "$(conda shell.bash hook)"

# set the path to the conda environments of your project.
export CONDA_ENVS_PATH="/home/yuhtong/scratch/andy/cd/envs"
export CONDA_PKGS_DIR="/home/yuhtong/scratch/andy/cd/pkgs"

conda activate base
conda activate comebin_env

output_dir="/home/yuhtong/scratch/andy/CAMI_datasets/Corgi_COMEBin_gsa_marine_s0"
contig_file="/home/yuhtong/scratch/andy/simulation_short_read/2018.08.15_09.49.32_sample_0/contigs/anonymous_gsa.fasta"
bam_file_path="/home/yuhtong/scratch/andy/simulation_short_read/2018.08.15_09.49.32_sample_0/bam" # dummy.
temperature=0.15
emb_szs_forcov=2048
emb_szs=2048
batch_size=250
n_views=6
num_threads=10

awk '/^>/ {if (seqlen) print seqlen; seqlen=0; next} {seqlen+=length($0)} END {print seqlen}' "$contig_file" | sort -rn > ${contig_file}_lengths.txt

# CAL N50
total_length=$(awk '{sum+=$1} END {print sum}' ${contig_file}_lengths.txt)
target_length=$(awk -v total="$total_length" 'BEGIN {cutoff=total/2; current=0} {current+=$1; if (current >= cutoff) {print $1; exit}}' ${contig_file}_lengths.txt)

    # N50
echo "N50: $target_length"
    # Check if N50 is greater than 10000 and set tau accordingly
if [ "$target_length" -gt 10000 ]; then
    temperature=0.07
else
    temperature=0.15
fi
echo "Tau(temperature): ${temperature}"

# bash /home/yuhtong/scratch/andy/Corgi_MLP_CL/run_corgi_comebin.sh
# cd ./Corgi_MLP_CL/COMEBin
# python main.py generate_aug_data --contig_file ${contig_file} \
#     --out_augdata_path ${output_dir}/data_augmentation \
#     --n_views ${n_views} --bam_file_path ${bam_file_path} --num_threads ${num_threads}


# module load GCCcore/11.3.0; module load Python/3.10.4;
# source /home/yuhtong/scratch/andy/binferno/.venv/bin/activate
# CHECKPOINT_BASE=/data/gpfs/projects/punim1293/rob/corgi-order-base.ckpt

# cd ${output_dir}/data_augmentation/

# for aug in $(ls -d aug*/); do
#     corgi --input $aug/sequences_$(basename $aug).fasta --checkpoint $CHECKPOINT_BASE \
#         --embeddings $aug/corgi_base_$(basename $aug).npy \
#         --min-length 200 --output-csv $aug/corgi_base_$(basename $aug).csv
# done

# #########################################################################################################
# Training.
##########################################################################################################
folder=${output_dir}/comebin_res
keyword="embeddings.tsv"
cd ./binferno/

# note: the GPU may be oom due to large batch size. 1024 -> 512.
if [ -d "$folder" ]; then
    echo "${output_dir}/comebin_res exists."
    count=$(find "$folder" -maxdepth 1 -type f -name "*$keyword*" | wc -l)
    echo "Number of files containing '$keyword' in the folder: $count"
    if [ "$count" -ne 2 ]; then
        echo "Running getting representation."
        python main.py train --data ${output_dir}/data_augmentation \
        --temperature ${temperature} --emb_szs_forcov ${emb_szs_forcov} \
        --batch_size ${batch_size} --emb_szs ${emb_szs} --n_views ${n_views} \
        --add_model_for_coverage \
        --output_path ${output_dir}/comebin_res --earlystop --addvars --vars_sqrt --num_threads ${num_threads}
    else
        echo "No need to run getting representation."
    fi
else
    echo "${output_dir}/comebin_res does not exist."
    echo "Running getting representation."
    python main.py train --data ${output_dir}/data_augmentation \
    --temperature ${temperature} --emb_szs_forcov ${emb_szs_forcov} \
    --batch_size ${batch_size} --emb_szs ${emb_szs} --n_views ${n_views} \
    --add_model_for_coverage \
    --output_path ${output_dir}/comebin_res --earlystop --addvars --vars_sqrt --num_threads ${num_threads}
fi

# binning

emb_file=${output_dir}/comebin_res/embeddings.tsv
seed_file=${contig_file}.bacar_marker.2quarter_lencutoff_1001.seed

python main.py bin --contig_file ${contig_file} \
    --emb_file ${emb_file} \
    --output_path ${output_dir}/comebin_res \
    --seed_file ${seed_file} --num_threads ${num_threads}

# ########################################################################################################
# # Get final result with CheckM evaluation
checkm_database=/home/yuhtong/scratch/andy/cd/envs/comebin_env/checkm_data
python main.py get_result --contig_file ${contig_file} \
    --output_path ${output_dir}/comebin_res \
    --seed_file ${seed_file} --num_threads ${num_threads} \
    --checkm_database ${checkm_database}

if [[ $? -ne 0 ]] ; then echo "Something went wrong with running clustering. Exiting.";exit 1; fi

