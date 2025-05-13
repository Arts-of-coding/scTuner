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
from sctuner.optimisers import AdEMAMix
from sctuner.pqutils import pqsplitter
from sctuner.pqutils import pqconverter
from sctuner.pqutils import pqmerger

# Import the pipe for the separate functions
from sctuner.pqutils import Parquetpipe

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

main_dir = "tests/pytestdata/"
subfolders = [ f'{main_dir}{f.name}/' for f in os.scandir(main_dir) if f.is_dir() ]
data_dir_list = subfolders

outdir = f"{main_dir}"
feature_file = f"{main_dir}features_scalesc_outer_joined.txt"

def test_parquet_output():
    args = [{},                                             # pqsplitter kwargs
            {"device":"cpu"},                               # pqconverter kwargs e.g. "dtype_raw":"UInt32"
            {"low_memory":True}]  #                         # pqmerger kwargs

    pqpipe = Parquetpipe(data_dir_list, feature_file, outdir)
    pqpipe.setup_parquet_pipe(*args)
    pqmerger(outdir)
    
    assert os.path.exists( "tests/pytestdata/joined_dataset.parquet") == 1
    assert os.path.exists( "tests/pytestdata/joined_dataset_raw.parquet") == 1