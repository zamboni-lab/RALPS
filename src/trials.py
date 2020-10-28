import pandas
import numpy as np
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.constants import data_path as path
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from src.torch import cl
import torch

if __name__ == "__main__":

    path = path.replace('data', 'res')

    _, X_test, _, y_test = cl.get_data(path + 'encodings.csv')

    print('creating new model')
    model = cl.Classifier(input_shape=100, n_batches=7)
    print('loading state_dict()')
    model.load_state_dict(torch.load(path + 'classifier.torch', map_location='cpu'))
    model.eval()

    predictions = model(torch.Tensor(X_test))
    true_positives = (predictions.argmax(-1) == torch.LongTensor(y_test)).float().detach().numpy()
    test_accuracy = true_positives.sum() / len(true_positives)
    print('achieved accuracy:', test_accuracy)