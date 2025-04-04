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

main_dir = "/mnt/d/Radboud/conda_environments/sctunerdevrebase/scTuner/pytestdata/" # 4080super 1mln
subfolders = [ f'{main_dir}{f.name}/' for f in os.scandir(main_dir) if f.is_dir() ]
data_dir_list = subfolders

outdir = f"{main_dir}"
feature_file = f"{main_dir}features_scalesc_outer_joined.txt"

# Try out with kwargs later...
args = [{},                                             # pqsplitter kwargs
        {},                                             # pqconverter kwargs e.g. "dtype_raw":"UInt32"
        {"low_memory":True}]  #                         # pqmerger kwargs

pqpipe = Parquetpipe(data_dir_list, feature_file, outdir)
pqpipe.setup_parquet_pipe(*args)
pqmerger(outdir)