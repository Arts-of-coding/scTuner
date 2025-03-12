import polars as pl
import scanpy as sc
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import torch
from torch.optim import Optimizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from sctuner.optimisers import AdEMAMix
import numpy as np
from sctuner.vae import VAE, loss_function, train
from torch.utils.data import DataLoader
import torch

# Inside model.SCVI.setup_anndata
def setup_parquet(parquet_path: str, metadata_columns: list = ["ID","sctuner_batch"], device: str = "cpu"):
    ''' Docstring 
    device: choices in "cpu" (default) or "gpu" (cuda). If there is enough VRAM and the dataset is small enough suggest to use "gpu". If there is a good amount of RAM "cpu" should be selected.
    '''

    cells = pl.scan_parquet(parquet_path)
    cells.collect_schema()

    match device:
        case "gpu":
            gpu_engine = pl.GPUEngine(
                device=0, # This is the default
                raise_on_fail=False,
                parquet_options={
                    'chunked': True,
                    'chunk_read_limit': int(1e2),
                    'pass_read_limit': int(4e9)
                } # Fail loudly if we can't run on the GPU.
            )

            result = (cells.collect(engine=gpu_engine)) # add gpu engine here!

        case "cpu":
            result = (cells.collect())

    result = result.drop(metadata_columns)
    print(result.head(3))

    # Perhaps save the list for the exact genes (or can use the earier)

    # Directly convert parquet to torch input
    result = result.to_torch()
    print(result.shape)

    # Scale the log1p values between 0 and 1 needed for pyTorch 
    result -= result.min(1, keepdim=True)[0]
    result /= result.max(1, keepdim=True)[0]

    return result


# Define a function to extract the embeddings
def extract_embeddings(model, x, device: str = "gpu"):
    ''' Save embeddings to gpu for numpy save. Gpu will be used by default to save normal RAM. '''

    if device == "gpu":
        device = "cuda"
        
    # Put embeddings on device of interest
    try: 
        x = x.to(device)
        z_mean, z_log_var = model.encode(x)

    except RuntimeError as e:
        print("Overriding device to 'cpu' due to too few VRAM available on gpu")
        device = "cpu"
        x = x.to(device)
        z_mean, z_log_var = model.cpu().encode(x)

    # match device if gpu allows for direct processing
    match device:
        case "gpu":
            z_mean = z_mean.cpu().detach().numpy()
        case "cpu":
            z_mean = z_mean.detach().numpy()
            
    return z_mean

