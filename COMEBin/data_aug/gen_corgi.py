import os
import shutil
import subprocess
import sys

# write a function running corgi via os.system or subprocess.
# This function only runs corgi once to embed one file.
def run_corgi(corgi_venv, input_fasta, min_length: int=1000, model_path: str=None):
    os.system(f"source {corgi_venv}") # run activate.
    CORGI_EXIST=shutil.which("corgi") # check if corgi is installed
    if CORGI_EXIST is None:
        sys.exit("Corgi is not installed in the specified conda environment, please install it first.")
    
    # Now run corgi embedding generation.
    output_emb = input_fasta.replace(".fasta", ".corgi_emb.npy")
    cmd = subprocess.run(["corgi",
                          "--input", input_fasta,
                          "--embeddings", output_emb,
                          "--min-length", str(min_length),
                          "--checkpoint", model_path], capture_output=True, text=True)
    if cmd.returncode != 0:
        sys.exit(f"Error in running Corgi:\n{cmd.stderr}")

    os.system(f"deactivate") # deactivate the conda environment
    return