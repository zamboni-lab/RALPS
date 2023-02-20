# RALPS
RALPS stands for Regularized Adversarial Learning Preserving Similarity. It's a method for eliminating batch effects in omics data, developed originally to harmonize multi-batch metabolomics measurements by MS. RALPS exploits reference samples (e.g. pooled study samples, NIST 1950 SRM, you name it...) present in every batch to assess interbatch differences. RALPS tries to reconstruct the original data while (i) removing interbatch differences on the basis of replicate measurements across batches and (ii) avoiding an expansion of the overall variance. 

In practice, **each batch should be first normalized individually to suppress intrabatch problems, e.g. temporal trends associated to drifts in LC-MS. RALPS is used in a second step to harmonize multiple batches.**

RALPS is particularly flexible in the experimental design. In fact, reference samples can be identical across all batches, but also vary between each pair of batches. In principle, it is also possible to include some samples from the previous one in the next batch and use these replicate measurements for training RALPS. 

RALPS preserves spectral properties and is robust against missing values.

RALPS includes a heuristic to automatically, identify the best set of parameters.

Principles and performance is described in the accompaining paper:
> Dmitrenko A, Reid M and Zamboni N, *Regularized adversarial learning for normalization of multi-batch untargeted metabolomics data*, Bioinformatics (2023), in press

## Requirements
```
hdbscan==0.8.27  
matplotlib==3.4.1  
numpy==1.20.0  
pandas==1.2.4  
scikit-learn==0.24.2  
scipy==1.6.3  
seaborn==0.11.1  
torch==1.8.1    
umap-learn==0.5.1
```
RALPS has been tested on CPU and GPU under MacOS and Windows.  
Training time required to normalize a dataset with ~3000 samples and ~150 metabolites was 5.82 minutes per run on average (30 epochs).

## How to run normalization

Run the following command from the `src` directory to normalize data with RALPS:  
`python ralps.py -n path/to/config.csv`

Config file should contain paths to the data and batch information files, and some other parameters.
All the required fields, as well as all the necessary parameters, are described below.  
Find the input example files [here](https://github.com/dmitrav/normalization/tree/master/examples).

### Config file structure

|       Parameter        |                                                           Comment                                                           |            Default value            |
|:----------------------:|:---------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------:|
|       data_path        |                                                   path to a csv data file                                                   |                  -                  |
|       info_path        |                                                path to a csv batch info file                                                |                  -                  |
|        out_path        |                                           path to a new folder to save results to                                           |                  -                  |
|       latent_dim       |                                          dimension of the bottleneck linear layer                                           | -1 (automatically derived from PCA) |
|     variance_ratio     |                                     percent of explained variance to derive latent_dim                                      |        0.9,0.95,0.99        |
|      n_replicates      |                                            mean number of replicates in the data                                            |                  3                  |
|       grid_size        |                                       size of the randomized grid search (# of runs)                                        |                  1                  |
|          d_lr          |                                                  classifier learning rate                                                   |            0.00005-0.005            |
|          g_lr          |                                                  autoencoder learning rate                                                  |            0.00005-0.005            |
|        d_lambda        |                                                    classifier loss coef                                                     |               0.-10.                |
|        g_lambda        |                                            autoencoder regularization term coef                                             |               0.-10.                |
|        v_lambda        |                                                     variation loss coef                                                     |               0.-10.                |
|      train_ratio       |                                                   train-test split ratio                                                    |                 0.9                 |
|       batch_size       |                                                   data loader batch size                                                    |              32,64,128              |
|         epochs         |                                                    # of epochs to train                                                     |                 30                  |
|      skip_epochs       |                                           # of epochs to skip for model selection                                           |                  3                  |
|    keep_checkpoints    |                                          save all model checkpoints after training                                          |    False (keep only best model)     |
|         device         |                                                 device to train on (Torch)                                                  |                 cpu                 |
|    plots_extension     |                                               save plots with this extension                                                |                 png                 |
| min_relevant_intensity | missing values before normalization are replaced with this;<br/>values below this after normalization are masked with zeros |                1000                 |
|  allowed_vc_increase   |                      fraction of sample's VC increase allowed (not contributing to the variation loss)                       |                0.05                 |

For most parameters, _coma separated values_ (e.g., `'batch_size'`) or _dash separated intervals_ (e.g., `'d_lr'`) can be provided.
For those, values will be uniformly sampled in the randomized grid search using defined options or intervals.
Otherwise, the exact values provided will be used.  
Default parameter values can be used by setting `'-1'`.

### Data file structure

|              |  sample_id_1  |  ...  | sample_id_M |
| :----------: | :--------:    | :--:  |  :--:       |
| feature_1    | count         |       |  count      |
| ...          |               |       |             |
| feature_N    | count         |       |  count      |


### Batch info file structure

|              |  batch     |  group  | benchmark |
| :----------: | :--------: |   :--:  |  :--:     |
| sample_id_1  | 1          |  reg_1  |  0        |
| sample_id_2  | 1          |  reg_1  |  0        |
| sample_id_3  | 2          |   0     |  0        |
| ...          |            |         |           |
| sample_id_M-1| k          |   0     |  bench_M  |
| sample_id_M  | k          |   0     |  bench_M  |

* __Batch__ column indicates samples' batch labels.  
* __Group__ column indicates groups of identical samples (replicates), _used for regularization_. 
If several samples have the same label (e.g., `'reg_1'`), they are treated as replicates of the same material.
While training, samples of the same group are encouraged to appear in the same cluster. Use `'0'` or `''` to provide no information about similarity of samples.
* __Benchmark__ indicates groups of identical samples taken as benchmarks in model evaluation. They are _not used for regularization_ while training, unless they appear in the group column as well.


## How to evaluate checkpoints

If you choose to keep checkpoints in the config file, you will find the autoencoder model at each training epoch saved in the `checkpoints` directory.
You can select a few checkpoints based on the training history to obtain alternative normalization solutions and the corresponding evaluation plots.  
To do that, remove unnecessary checkpoints and run the following command from the `src` directory:  
`python ralps.py -e path/to/directory/with/checkpoints/`

__Important:__ This works only with default RALPS output (directories and filenames should not be changed).  

## How to remove outliers 

If you wish to remove outliers from the normalized data, as proposed in the paper, run the following command from the `src` directory:  
`python ralps.py -r path/to/normalized/data.csv`

__Important:__ This works only with default RALPS output (directories and filenames should not be changed).

## How to change default configuration

If you wish to reconfigure RALPS (e.g., to use a different clustering algorithm as default, or to change default parameter values), you can do so by editing `src/constants.py`.
