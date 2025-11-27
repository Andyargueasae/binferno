import typer
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import pyfastx
import logging

app = typer.Typer()
from binferno.binferno_version import __version__ as ver

logger = logging.getLogger("Binferno\t" + ver)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(message)s")
console_hdr = logging.StreamHandler()
console_hdr.setFormatter(formatter)

class Binferno(object):
    """Binferno: A tool for binning metagenomic contigs using deep learning."""
    def __init__(self):
        # initialize logger.
        from binferno.binferno_version import __version__ as ver

        logger = logging.getLogger("Binferno\t" + ver)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(message)s")
        console_hdr = logging.StreamHandler()
        console_hdr.setFormatter(formatter)
    @app.command()
    def gen_aug(output_path: str, contigs: str, bam_dir: str, threads: int = 10, 
                min_contig_length: int = 1000, n_views: int = 6, corgi_model: str=None, plastid: bool = False):
        out_augdata_path = output_path
        os.makedirs(output_path, exist_ok=True)

        handler = logging.FileHandler(os.path.join(output_path, "binferno.log"))
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        from binferno.data_aug.generate_augfasta_and_saveindex import run_gen_augfasta
        from binferno.data_aug.gen_cov import run_gen_cov
        from binferno.data_aug.gen_var import run_gen_cov_var
        # from corgi import Corgi

        # modify the functions to take exact parameters rather than args.
        run_gen_augfasta(logger, contig_file=contigs, out_augdata_path=out_augdata_path,
                    contig_len=min_contig_length, n_views=n_views, model_path=corgi_model)
        run_gen_cov(logger, out_augdata_path=out_augdata_path, contigs=contigs, bam_file_path=bam_dir,
                    threads=threads, min_contig_length=min_contig_length, n_views=n_views, plastid=plastid)
        run_gen_cov_var(logger, out_augdata_path=out_augdata_path, num_threads=threads, n_views=n_views, 
                    contig_len=min_contig_length)

        return
    
    @app.command()
    def train(data_path: str, output_path: str, epochs: int = 200, batch_size: int = 1024, lr: float = 0.001,
                weight_decay: float = 1e-4, out_dim: int = 128, dropout: float = 0.2,
                emb_szs: int = 2048, n_layers: int = 3, emb_szs_forcov: int = 2048, out_dim_forcov: int = 128,
                n_layers_forcov: int = 3, log_every_n_steps: int=20, temperature: float = 0.1, n_views: int=6,
                fp16_precision: bool = False, cov_meannormalize: bool = True, cov_maxnormalize: bool = False,
                cov_standardization: bool = False, corgi_l2_normalize: bool = True, threads: int = 10, early_stop: bool = False,
                workers: int = 4, notuseaug0: bool = False, addvars: bool = False, vars_sqrt: bool = False, 
                add_model_for_coverage: bool = False):
        # training function for COMEBin.
        # resume the logger created from gen_aug.
        handler = logging.FileHandler(os.path.join(output_path, "binferno.log"))
        logger.addHandler(handler)
        logger.info("train")

        # pack-up every parameter into a dictionary and prepare it for saving the config file into yaml format.
        arg_dict = {
            "data_path": data_path,
            "output_path": output_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "out_dim": out_dim,
            "dropout": dropout,
            "emb_szs": emb_szs,
            "n_layers": n_layers,
            "emb_szs_forcov": emb_szs_forcov,
            "out_dim_forcov": out_dim_forcov,
            "n_layers_forcov": n_layers_forcov,
            "log_every_n_steps": log_every_n_steps,
            "temperature": temperature,
            "n_views": n_views,
            "fp16_precision": fp16_precision,
            "cov_meannormalize": cov_meannormalize,
            "cov_maxnormalize": cov_maxnormalize,
            "cov_standardization": cov_standardization,
            "corgi_l2_normalize": corgi_l2_normalize,
            "threads": threads,
            "early_stop": early_stop,
            "workers": workers,
            "notuseaug0": notuseaug0,
            "addvars": addvars,
            "vars_sqrt": vars_sqrt,
            "add_model_for_coverage": add_model_for_coverage
        }

        from train_CLmodel import train_CLmodel
        train_CLmodel(logger, device=None, num_threads=threads, output_path=output_path, notuseaug0=notuseaug0,
                data=data_path,n_views=n_views, workers=workers, batch_size=batch_size, epochs=epochs, lr=lr,
                weight_decay=weight_decay, out_dim=out_dim, dropout=dropout,
                emb_szs=emb_szs, n_layers=n_layers, emb_szs_forcov=emb_szs_forcov,
                out_dim_forcov=out_dim_forcov, n_layers_forcov=n_layers_forcov,
                log_every_n_steps=log_every_n_steps, temperature=temperature,
                fp16_precision=fp16_precision, cov_meannormalize=cov_meannormalize,
                cov_maxnormalize=cov_maxnormalize, cov_standardization=cov_standardization,
                corgi_l2_normalize=corgi_l2_normalize, early_stop=early_stop, addvars=addvars, vars_sqrt=vars_sqrt,
                add_model_for_coverage=add_model_for_coverage, arg_dict=arg_dict)
        return
    
    @app.command()
    def bin(contig_path: str, seed_file: str, emb_file: str, output_path: str, not_l2normalize: bool = False,
            contig_len: int = 1000, num_threads: int = 10, plastid: bool = False):
        # resume the logger first.
        handler = logging.FileHandler(os.path.join(output_path, "binferno.log"))
        logger.addHandler(handler)
        logger.info("binning")
        from utils import gen_seed
        from cluster import cluster
        if plastid:
            _ = gen_seed(
                logger,
                contig_path,
                num_threads,
                contig_len,
                marker_name="pltd_marker",
                quarter="2quarter",
            )
        else:
            _ = gen_seed(
                logger,
                contig_path,
                num_threads,
                contig_len,
                marker_name="bacar_marker",
                quarter="4quarter",
            )
        cluster(logger, emb_file=emb_file, seed_file=seed_file, output_path=output_path, contig_file=contig_path, contig_len=contig_len, 
            not_l2normaize=not_l2normalize, cluster_num=None, num_threads=num_threads)
        return
    
    @app.command()
    def get_results(contig_file: str, output_path: str, num_threads: int = 10, 
            bac_mg_table: str = None, ar_mg_table: str = None, pltd_mg_table: str = None, 
            plastid: bool = False, min_binsize: int=200000, checkm_database: str = None, contig_len: int = 1000):
        # resume the logger first.
        handler = logging.FileHandler(os.path.join(output_path, "binferno.log"))
        logger.addHandler(handler)
        logger.info("Retrieving results")
        from utils import gen_seed
        from get_final_result import run_get_final_result

        if plastid:
            seed_num = gen_seed(
                logger,
                contig_file,
                num_threads,
                contig_len,
                marker_name="pltd_marker",
                quarter="2quarter",
            )
        else:
            seed_num = gen_seed(
                logger,
                contig_file,
                num_threads,
                contig_len,
                marker_name="bacar_marker",
                quarter="2quarter",
            )

        run_get_final_result(
            logger, bac_mg_table,
            ar_mg_table,
            pltd_mg_table,
            checkm_database,
            contig_file,
            plastid,
            output_path,
            min_binsize,
            seed_num, 
            num_threads,
            ignore_kmeans_res=True)

        return

if __name__ == "__main__":
    app()
