import pandas, numpy, os, sys, umap
from src.preprocessing import run_pca

if __name__ == "__main__":

    path = '/Users/andreidm/ETH/projects/normalization/res/grid_search/'

    for folder in os.listdir(path):
        if not folder.startswith('.'):
            cps = os.listdir(path + folder + '/checkpoints/')
            if len(cps) == 1:
                continue
            else:
                min_name_len = min([len(cp) for cp in cps])

                for checkpoint in cps:
                    if len(checkpoint) == min_name_len:
                        continue
                    else:
                        os.remove(path + folder + '/checkpoints/' + checkpoint)

