from __future__ import print_function, division
import pandas as pd
import numpy as np
import tensorflow as tf
import h5py


DATA = 'C:\\Users\\rotem\PycharmProjects\ML4Jets\Data\events_anomalydetection.h5'
# DATA = '/Users/rotemmayo/Documents/PyCharm/ML4Jets-HUJI/jets_data_tiny.h5'

N_EVENTS = 100000
IS_JETS = False
TEST_PERCENT = 0.2
DATA_SET_NAME = 'dataset_1'
BATCH_SIZE = 128
EPOCHS = 10
DROPOUT = 0.1
LEARNING_RATE = 10**-2

"""
Option 1: Load everything into memory
"""


def read_h5():
    if IS_JETS:
        df = h5py.File(DATA, 'r').get(DATA_SET_NAME)
    else:
        df = pd.read_hdf(DATA, stop=N_EVENTS)
        tf.keras.utils.normalize(df, axis=-1, order=2)
        print("Memory in GB:", sum(df.memory_usage(deep=True)) / (1024 ** 3))
    print(df.shape)
    return df


"""
# Option 3: Use generator to loop over the whole file
"""


def generator(filename, chunksize=512, total_size=1100000):
    i = 0
    while True:
        yield pd.read_hdf(filename, start=i * chunksize, stop=(i + 1) * chunksize)
        i += 1
        if (i + 1) * chunksize > total_size:
            i = 0


"""
Creates a tensorflow data set from a pandas dataframe, assuming the target value is in the last column.
"""


def create_dataset_jets(df):
    target = df[:, 0].astype(int)
    df = df[:, 1:]
    dataset = tf.data.Dataset.from_tensor_slices((df, target))
    for feat, targ in dataset.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))
    shuffled_dataset = dataset.shuffle(len(df)).batch(BATCH_SIZE)
    return shuffled_dataset


"""
Creates a tensorflow data set from a pandas dataframe, assuming the target value is in the last column.
"""


def create_dataset(df):
    target = df.iloc[:, -1].astype(int)
    df = df.iloc[:, :-1]
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values)).batch(BATCH_SIZE)
    for feat, targ in dataset.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))
    shuffled_dataset = dataset.shuffle(len(df)).batch(1)
    return shuffled_dataset


"""
TODO: Add proper initialization
"""


def learn(df):
    msk = np.random.rand(len(df)) > TEST_PERCENT
    if IS_JETS:
        train = df[msk, :]
        test = df[~msk, :]
        test_dataset = create_dataset_jets(test)
        train_dataset = create_dataset_jets(train)
    else:
        train = df[msk]
        test = df[~msk]
        test_dataset = create_dataset(test)
        train_dataset = create_dataset(train)
    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.fit(train_dataset, epochs=EPOCHS, verbose=2)
    model.evaluate(test_dataset)


def main():
    # gen = generator(DATA)
    df = read_h5()
    learn(df)


main()
"""
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
"""