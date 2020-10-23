
import pandas, numpy
from src.constants import data_path as path

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    data = pandas.read_csv(path.replace('data', 'res') + "samples_encodings.csv")
    sample_names = data.iloc[:,0]

    # load dataset
    X = data.values[:, 2:].astype(float)
    Y = data.values[:, 1].astype(int) - 1  # batches from 0 to 6
    Y = to_categorical(Y)

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)

    input = Input(shape=(100,))
    x = Dense(7, activation='relu')(input)
    output = Dense(7, activation='softmax')(x)
    model = Model(input, output)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, epochs=100, batch_size=128, shuffle=True, validation_data=(X_val, y_val))

    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)

    print('\nAccuracy:', test_acc)
