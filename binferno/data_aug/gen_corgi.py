from corgi import Corgi

# app = typer.Typer()
base_model="/data/gpfs/projects/punim1293/rob/corgi-order-base.ckpt"
plastid_plus="/data/scratch/projects/punim1293/andy/corgi-July30-plastidplus.ckpt"

def run_corgi(input_fasta, output_csv: str=None, 
              min_length: int=1000, 
              model_path: str=plastid_plus, 
              embedding: str=None):
    corgi = Corgi()
    prediction = corgi.predict(input=[input_fasta], output_csv=output_csv, embeddings=embedding, min_length=min_length, checkpoint=model_path)
    return

# if __name__ == "__main__":
#     app()