# RALPS
RALPS stands for Regularized Adversarial Learning Preserving Similarity of samples.
It's a novel method for eliminating batch effects in omics data, developed originally to harmonize distant-in-time multi-batch flow-injection mass-spectrometry measurements.

<img src="https://github.com/dmitrav/normalization/blob/v6_dev/schematic/figure.png" alt="RALPS" width="600"/>

I am currently working towards publishing it.

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

## How to run

You can run RALPS from command line by typing:  
`python ralps.py path/to/config.csv`

Config file contains paths to the data and batch information files, as well as other parameters.

### Config file structure

|       Parameter        |                                     Comment                                      |              Default value              |
|:----------------------:|:--------------------------------------------------------------------------------:|:---------------------------------------:|
|       data_path        |                             path to a csv data file                              |                    -                    |
|       info_path        |                          path to a csv batch info file                           |                    -                    |
|        out_path        |                       path to a folder to save results to                        |                    -                    |
|       latent_dim       |                          dim of bottleneck linear layer                          | -1 (automatically derived based on PCA) |
|     variance_ratio     |                percent of explained variance to derive latent_dim                |          0.7,0.8,0.9,0.95,0.99          |
|      n_replicates      |                      mean number of replicates in the data                       |                    3                    |
|       grid_size        |              size of grid search (number of RALPS runs to perform)               |                    1                    |
|          d_lr          |                             classifier learning rate                             |              0.00005-0.005              |
|          g_lr          |                            autoencoder learning rate                             |              0.00005-0.005              |
|        d_lambda        |                           classifier regularizer coef                            |                 0.-10.                  |
|        g_lambda        |                           autoencoder regularizer coef                           |                 0.-10.                  |
|        v_lambda        |                               variation loss coef                                |                 0.-10.                  |
|      train_ratio       |                              train-test split ratio                              |                   0.9                   |
|       batch_size       |                              data loader batch size                              |                32,64,128                |
|         epochs         |                                n epochs to train                                 |                   30                    |
|      skip_epochs       |                       n epochs to skip for model selection                       |                    3                    |
|    keep_checkpoints    |                    save all model checkpoints after training                     |      False (keep only best model)       |
|         device         |                                device to train on                                |                   cpu                   |
|    plots_extension     |                          save plots with this extension                          |                   png                   |
| min_relevant_intensity |                    data values below will be masked with this                    |                  1000                   |
|  allowed_vc_increase   | percent of sample's VC increase allowed (not contributing to the variation loss) |                  0.05                   |

For most parameters, _coma separated values_ (e.g., batch_size) or _dash separated intervals_ (e.g., d_lr) can be provided.
For those parameters values will be _uniformly sampled_ during the grid search, using supplied options / intervals.
Otherwise, the exact values provided will be used. _Default_ parameter values can be used by just setting __-1__.

### Data file structure

|              |  sample_id_1  |  ...  | sample_id_M |
| :----------: | :--------: | :--:  |  :--:    |
| metabolite_1 | count      |       |  count   |
| ...          |            |       |          |
| metabolite_N | count      |       |  count   |


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
If several samples have the same label (e.g., 'reg_1'), they are treated as replicates of the same material.
While training, samples of the same group are encouraged to appear in the same cluster. Use 0 to provide no information about similarity of samples.
* __Benchmark__ indicates groups of identical samples taken as benchmarks in model evaluation. They are _not used for regularization_ while training.

### Examples
Find input file examples [here](https://github.com/dmitrav/normalization/tree/master/examples).
