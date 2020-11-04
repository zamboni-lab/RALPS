import pandas, numpy, os, sys
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.constants import data_path as path

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import torch

if __name__ == "__main__":

    print(sys.argv[1])