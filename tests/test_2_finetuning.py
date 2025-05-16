import polars as pl
import polars.selectors as cs
import scanpy as sc
import numpy as np
import random
import os
import anndata as ad
import time
from tqdm import tqdm
import math
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.distributions import Normal
import torch.nn as nn
import torch.optim as optim
import glob
from memory_profiler import profile

# Import functions from package separate
import sctuner as sct

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

main_dir = "tests/pytestdata/1/"
subfolders = [ f'{main_dir}{f.name}/' for f in os.scandir(main_dir) if f.is_dir() ]
data_dir_list = subfolders

outdir = f"{main_dir}"

# Using file constructed during biosb2025 demo
feature_file = "tests/pytestdata/features_scalesc_outer_joined.txt"

def test_finetuning_pq_output():
    args =  [{"suffix":"scanpy_hvg_object.h5ad"},           # pqsplitter kwargs
            {"device":"cpu"},                               # pqconverter kwargs e.g. "dtype_raw":"UInt32"
            {}]  #                                          # pqmerger kwargs

    pqpipe = sct.pqutils.Parquetpipe(data_dir_list, feature_file, outdir)
    pqpipe.setup_parquet_pipe(*args)

    assert os.path.exists("tests/pytestdata/1/joined_dataset.parquet") == 1
    assert os.path.exists("tests/pytestdata/1/joined_dataset_raw.parquet") == 1