# Testing scalesc on 2GB ram videocard with the 1mln dataset (3x 100k version)
import scanpy as sc
import numpy as np
import pandas as pd
import importlib
import scalesc as ssc
from tqdm import tqdm
from memory_profiler import profile
import random
import os
import anndata as ad
import time
from sctuner.scalesc import hvg_batch_processing, extract_hvg_h5ad

# also needed this extra: uv pip install nvidia-cuda-nvcc-cu11
# check drivers with sudo apt update && upgrade
# nvidia-smi should display your GPU

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print(f'func:{f.__name__} args:[{args}, {kw}] took: {te-ts:.4f} sec')
        return result

    return timed

# Can also load in multiple anndata objects; but loses all obs? and makes an inner join; important!
# Concatenate and calculate HVGs for each batch
import os 
main_dir = "/mnt/d/Radboud/data/python/jupyter_notebooks/drvi_pbmc/"

subfolders = [ f'{main_dir}{f.name}/' for f in os.scandir(main_dir) if f.is_dir() ]
data_dir_list = subfolders

hvg_batch_processing(data_dir_list)

feature_file = f"{main_dir}features_scalesc_outer_joined.txt"

extract_hvg_h5ad(data_dir_list, feature_file)