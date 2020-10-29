import pandas
import numpy as np
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

    sample_types = ["_".join(x.split('_')[:2]) for x in shared_perturbations]
    sample_types = set(sample_types)
    print(sample_types)
