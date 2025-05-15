# Testing scalesc on 2GB ram videocard with the 1mln dataset (3x 100k version)
import scalesc as ssc
from tqdm import tqdm
#from memory_profiler import profile
import random
import os

def hvg_batch_processing(dirs: list,top_batch_hvg: int = 1500, preliminary_filtering: bool = True):
    ''' Put function arguments here'''
    # calculate hvg for each batch
    for _ in tqdm(dirs):

        # Load in datasets separately
        scalesc = ssc.ScaleSC(data_dir=_, 
                        max_cell_batch=1e4, # no more than 1e5 is suggested
                        preload_on_cpu=True, 
                        preload_on_gpu=True, 
                        save_raw_counts=False,
                        save_norm_counts=False,
                        output_dir=f'scalesc_output{_}')
        print(scalesc.adata.shape)

        if preliminary_filtering is True:
            scalesc.calculate_qc_metrics()
            scalesc.filter_genes(min_count=3) # this gives an error, but does not have to be a problem perhaps?
            scalesc.filter_cells(min_count=200, max_count=6000)
            print('shape after preliminary filtering', scalesc.adata.shape)

        # hvg
        scalesc.highly_variable_genes(n_top_genes=top_batch_hvg) # Can do batch key here and just select + save afterwards...
        list_hvgs = scalesc.adata.var[scalesc.adata.var['highly_variable']]
        list_hvgs

        # Save features
        features = list_hvgs.index.to_list()

        # Randomly shuffle the features for model training later
        import random
        random.seed(0)
        features = random.sample(features, len(features))

        # Write features as separate list into original data_dir
        with open(f"{_}features_scalesc.txt", "w") as output:
            output.write(str(features))

        del list_hvgs,features
        del scalesc


def extract_hvg_h5ad(dirs: list, feature_file_path: str, join: str = "outer"):

    random.seed(0)
    res = []

    for num, _ in enumerate(dirs):

        with open(f"{_}features_scalesc.txt", 'r') as f:
            text = f.read()
            features = eval(text)

            match join:
                case "inner":
                    res = list(set(res).intersection(features)) if num > 0 else features
                case "outer":
                    res = list(set(res).union(features)) if num > 0 else features
        
    res = random.sample(res, len(res))

    # Write features as separate list into original data_dir
    with open(feature_file_path, "w") as output:
        output.write(str(res))
        
    print("Joining on HVGs: ",len(res))

    for _ in tqdm(dirs):

        if not os.path.exists(f"{_}scanpy_hvg_out/"):
            os.makedirs(f"{_}scanpy_hvg_out/")

        scalesc = ssc.ScaleSC(data_dir=_, 
                        max_cell_batch=1e4, # no more than 1e5 is suggested
                        preload_on_cpu=False,  #True
                        preload_on_gpu=True, 
                        save_raw_counts=False,
                        save_norm_counts=False,
                        output_dir=f'scalesc_output{_}')
        print(scalesc.adata.shape)
        print(scalesc.adata.var.index)

    # Subset with HVGs taking count data when present
        res = list(set(res).intersection(scalesc.adata.var.index))

        scalesc.adata[:,res] #
        scalesc.to_CPU()

        adata = scalesc.adata_X
        adata.write_h5ad(f"{_}scanpy_hvg_out/scanpy_hvg_object.h5ad")
