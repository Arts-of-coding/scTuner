import polars as pl

# Inside model.SCVI.setup_anndata
def setup_parquet(parquet_path: str, feature_file_path: str , metadata_columns: list = ["ID","sctuner_batch"], device: str = "cpu", finetuning: bool = False):
    ''' Docstring 
    device: choices in "cpu" (default) or "gpu" (cuda). If there is enough VRAM and the dataset is small enough suggest to use "gpu". If there is a good amount of RAM "cpu" should be selected.
    finetuning: choices in True or False (default). If True, the model will be finetuned on the dataset and zero imputation for missing genes will be applied.
    '''

    cells = pl.scan_parquet(parquet_path, low_memory=True) #
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
            result = (cells.collect()) #
            #result = pl.read_parquet(parquet_path, streaming=True) # add gpu engine here!

    result = result.drop(metadata_columns)

    # Load in features
    with open(feature_file_path, 'r') as f:
        text = f.read()
        features = eval(text)

    # Add missing columns if there are any
    if finetuning is True:
        
        for i in features:
            if i not in result.columns:
                result = result.with_columns(pl.lit(0).alias(i))

    # Order the dataframe by the features
    result = result.select(features)
	
    print(result.head(3))

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

    except RuntimeError:
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

