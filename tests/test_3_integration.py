import polars as pl
import scanpy as sc
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import torch
import os
from torch.optim import Optimizer
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Import scTuner functions and modules
import sctuner as sct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_dir = "tests/pytestdata/1/"

# Define the input parquet file (already log1p normalized)
path_parquet = 'tests/pytestdata/1/joined_dataset.parquet'
feature_file = "tests/pytestdata/features_scalesc_outer_joined.txt"

def test_training():
    result = sct.models.setup_parquet(parquet_path=path_parquet, feature_file_path=feature_file)

    # Load in the dataset into PyTorch's DataLoader
    train_loader_train = DataLoader(result, batch_size=512, shuffle=True, pin_memory=True, num_workers=1)

    # Initialize the model and optimizer
    model = sct.vae.VAE(input_dim=result.shape[1]).to(device)
    optimizer_AdEMAMix = sct.optimisers.AdEMAMix(model.parameters())

    # Train the model (VAE from scTuner)
    sct.vae.train(model = model, optimizer = optimizer_AdEMAMix, train_loader = train_loader_train, epochs=10, device=device)

    # Extract and save the embeddings back to the cpu (often needed for large datasets due to insufficient VRAM)
    embeddings = sct.models.extract_embeddings(model, result, device="cpu")

    # Save embeddings & model
    np.save("embeddings", embeddings)
    torch.save(model, "model_vae.pt")

    # Convert everything to a standard single-cell object (containing raw counts) for further downstream analysis
    adata, embeddings = sct.pqutils.parquet2anndata(f'{main_dir}joined_dataset_raw.parquet', embeddings_path="embeddings.npy", outputfile_path=f"{main_dir}adata_raw_embeddings.h5ad")
    assert os.path.exists(f"{main_dir}adata_raw_embeddings.h5ad") == 1


