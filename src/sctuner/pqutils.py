import polars as pl
import scanpy as sc
from tqdm import tqdm
import torch
import numpy as np
import os
import polars.selectors as cs
import glob
import shutil
from memory_profiler import profile
import contextlib
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pqsplitter(dirs: list, feature_file_path: str, batch_size: int = 50000, suffix: str = "scanpy_hvg_out/scanpy_hvg_object.h5ad",outputdir: str = "sctuner_output/data/"):
    ''' This function converts scanpy objects to parquet files in a batch specific manner (CPU). '''
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    with open(feature_file_path, 'r') as f:
        text = f.read()
        features = eval(text)

    data_dir_list = []
    # read in data directories:
    for _ in dirs:
        data_dir_list.append(f'{_}{suffix}')

    print(data_dir_list)
    batch_len = batch_size

    for num, i in enumerate(data_dir_list):
        # Write data in batches for parquet conversion
        adata = sc.read_h5ad(i)
        ceil_val = np.ceil(len(adata.obs)/batch_len).astype(int)
        genes = list(set(features).intersection(adata.var.index))
        start = 0
        end = batch_len

        for _ in tqdm(range(0, ceil_val)):
            dataset_split = adata[start:end,]
            cond = f"dataset_{num}_batch_{_}"
            dataset_split.obs["sctuner_batch"] = cond
            dataset_split.obs["ID"] = dataset_split.obs.index.tolist()

            plotdf = sc.get.obs_df(
                    dataset_split,
                    keys=[*genes,"sctuner_batch","ID"], #*s_genes,*g2m_genes
                )
            plotdf.to_parquet(f"{outputdir}dataset_{num}_split_{_}.parquet")
            print(plotdf.shape)
            start += batch_len
            end += batch_len
            del plotdf
        del adata


def pqconverter(dirs: list, feature_file_path: str, batch_size_genes: int = 100, outputdir: str = "sctuner_output/data/", dtype_raw: str = "UInt16", device: str = "gpu"):
    ''' This function converts parquet files from pandas to polars and performs log1p normalisation on the individual parquet files (GPU-accelerated). '''
    # Define the gpu_engine
    gpu_engine = pl.GPUEngine(
        device=0, # This is the default
        raise_on_fail=True,
        parquet_options={
            'chunked': False,
            'chunk_read_limit': int(1e9),
            'pass_read_limit': int(4e9)
        } # Fail loudly if we can't run on the GPU.
    )

    # Load in features
    with open(feature_file_path, 'r') as f:
        text = f.read()
        features = eval(text)

        kwargs = dict(
        use_pyarrow=True,
        pyarrow_options=dict(partition_cols=["sctuner_batch"])
    )

    print(len(features))

    # Remove any old directories if present
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(f'{outputdir}joined_dataset_raw/')

    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(f'{outputdir}joined_dataset/')

    # Define polars engine based on gpu availability
    match device:
        case "gpu":
            try: 
                gpu_engine = pl.GPUEngine(
                    device=0, # This is the default
                    raise_on_fail=True,
                    parquet_options={
                        'chunked': True,
                        'chunk_read_limit': int(1e2),
                        'pass_read_limit': int(4e9)
                    } # Fail loudly if we can't run on the GPU.
                )
                engine_polars = gpu_engine
            except RuntimeError:
                print("GPU not available, using CPU instead.")
                engine_polars = "auto"

        case "cpu":
            engine_polars = "auto"

    for num, i in enumerate(dirs):

        for _ in tqdm(range(0, len(glob.glob(f'{outputdir}dataset_{num}*.parquet')))):
            file = f"{outputdir}dataset_{num}_split_{_}.parquet"
            df = pl.scan_parquet(file).collect(engine=engine_polars)

            # Cast missing colums for correct schema
            for i in features:
                if i not in df.columns:
                    df = df.with_columns(pl.lit(0).alias(i))
            
            # Matches the datatype to the dtype U32 for datasets with large values
            match dtype_raw:
                case "UInt16":
                    df = df.cast({cs.numeric(): pl.UInt16})
                case "UInt32":
                    df = df.cast({cs.numeric(): pl.UInt32})

            df.with_columns(pl.col('ID').cast(pl.String))
            df.with_columns(pl.col('sctuner_batch').cast(pl.String))

            # Specifying the order and schema of the parquet file splits
            schema_order = ["ID","sctuner_batch",*features]

            df = df.select(schema_order)
            print(df.shape)

            # Write raw parquet files batch wise
            df.write_parquet(f'{outputdir}/joined_dataset_raw', **kwargs)

            # Scale with log1p beforehand for faster training!
            IDs_result_datasets = df.select(pl.col('ID'),pl.col('sctuner_batch'))
            IDs_result_datasets

            df = df.drop(["ID","sctuner_batch"])

            # Convert genes in a batch specific way for log1p scaling
            batch_len = batch_size_genes
            start = 0
            end = batch_len

            with open(feature_file_path, 'r') as f:
                text = f.read()
                features = eval(text)

            ceil_val = np.ceil(len(features)/batch_len).astype(int)

            for _ in tqdm(range(0,ceil_val)):
                features_sub = features[start:end]
                df = df.with_columns(pl.col(features_sub).log1p().round(1))

                start += batch_len
                end += batch_len

            df = df.cast({cs.numeric(): pl.Float32})
            df = df.with_columns(IDs_result_datasets)
            print(df.head(3))

            # Write log1p transformed parquet files batch wise
            df.write_parquet(f'{outputdir}/joined_dataset', **kwargs)
            del df


def pqmerger(outputdir: str = "sctuner_output/data/"): #, **kwargs_polars
    ''' This function joins the raw count parquet files and the log1p normalised ones into a single object (CPU). '''

    # non-raw joined
    cells = pl.scan_parquet(f'{outputdir}joined_dataset/', low_memory=True) # ,**kwargs_polars
    cells.collect_schema()

    result = (cells.collect())
    print(result.shape)
    result = result.sample(n=len(result), seed=42,shuffle=True)

    print(result.shape)
    print(result.head(5))
    result.write_parquet(f'{outputdir}joined_dataset.parquet', use_pyarrow=True)
    del cells
    del result

    # Remove temporary splits if join is succesfull
    shutil.rmtree(f'{outputdir}joined_dataset/')

    # raw joined
    cells = pl.scan_parquet(f'{outputdir}joined_dataset_raw/', low_memory=True)
    cells.collect_schema()

    result = (cells.collect())
    print(result.shape)
    result = result.sample(n=len(result), seed=42,shuffle=True)
    
    print(result.shape)
    print(result.head(5))

    result.write_parquet(f'{outputdir}joined_dataset_raw.parquet')

    # Remove temporary splits if join is succesfull
    shutil.rmtree(f'{outputdir}joined_dataset_raw/')
    del result
    del cells


def parquet2anndata(parquet_path: str, embeddings_path: str, metadata_columns: list = ["ID","sctuner_batch"], device: str = "cpu", outputfile_path: str = "adata_raw_embeddings.h5ad", write_output: bool = True, **kwargs_polars):
    ''' 
    device: choices in "cpu" (default) or "gpu" (cuda). If there is enough VRAM and the dataset is small enough suggest to use "gpu". If there is a good amount of RAM "cpu" should be selected.
    '''

    cells = pl.scan_parquet(parquet_path, **kwargs_polars)
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


    dict_columns = {}

    for _ in metadata_columns:
        dict_columns[_] = result[_]

    result = result.drop(metadata_columns)

    cols = result.columns
    merged_numpy = result.to_numpy()

    del result

    adata = sc.AnnData(merged_numpy)
    del merged_numpy

    for _ in metadata_columns:
        adata.obs[_] = dict_columns[_]

    adata.var.index = cols

    # Load in embeddings
    print("Adding embeddings")
    embeddings = np.load(embeddings_path)

    for i, z in enumerate(embeddings.T): # remove .cpu() if on cpu
        adata.obs[f"Z_{i}"] = z

    # Write h5ad with embeddings attached
    if write_output is True:
        print("Write h5ad object")
        sc.write(outputfile_path,adata)

    return adata, embeddings

class Parquetpipe:
    def __init__(self, dirs, feature_file_path, outputdir):
        self.dirs = dirs
        self.feature_file_path = feature_file_path
        self.outputdir = outputdir
        #for key, value in kwargs.items():
        #    setattr(self, key, value)
    @profile
    def setup_parquet_pipe(self, *args): # Can add configurable gpu_engine as well as possible second argument
        splitter_params = args[0]
        converter_params = args[1]
        #merger_params = args[2]
        self.__dict__.update(**splitter_params)
        self.__dict__.update(**converter_params)
        #self.__dict__.update(**merger_params)

        pqsplitter(dirs = self.dirs, feature_file_path = self.feature_file_path, outputdir = self.outputdir, **splitter_params)
        pqconverter(dirs = self.dirs, feature_file_path = self.feature_file_path, outputdir = self.outputdir, **converter_params)
        #pqmerger(outputdir = self.outputdir) #, **merger_params

        return