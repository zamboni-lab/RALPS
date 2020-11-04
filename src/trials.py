import pandas, numpy, os
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.constants import data_path as path
from src.constants import shared_perturbations
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from src.torch import cl
import torch

if __name__ == "__main__":

    path = '/Users/andreidm/ETH/projects/normalization/res/'

    new = path + 'test_folder/test_folder'

    if not os.path.exists(new):
        os.makedirs(new)