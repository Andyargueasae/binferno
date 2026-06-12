import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from binferno.training import SimCLRTrainer


'''
Mainly tests the SimCLR path: from embeddings to dataloading to training and encoding. 
Create dummy data to serve testing purposes.  
'''
def dummy_args_simclr():
    return {
        "data_dir": Path("dummy_data"),
        "output_dir": Path("dummy_output"),
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "num_workers": 0,
    }

def dummy_args_barlowtwins():
    return {
        "data_dir": Path("dummy_data"),
        "output_dir": Path("dummy_output"),
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "num_workers": 0,
    }

def test_simclr_train_an_epoch():
    args = dummy_args_simclr()
    trainer = SimCLRTrainer(args)
    trainer.train()

def prep_model_simclr():
    args = dummy_args_simclr()
    trainer = SimCLRTrainer(args)
    trainer.train()
    return trainer

def test_barlowtwins_train_an_epoch():
    args = dummy_args_barlowtwins()
    trainer = SimCLRTrainer(args)
    trainer.train()

def prep_model_barlowtwins():
    args = dummy_args_barlowtwins()
    trainer = SimCLRTrainer(args)
    trainer.train()
    return trainer