from __future__ import print_function, division
import pandas as pd
import numpy as np
import tensorflow as tf
import h5py
import os


"""
NOTE: jets_data_tiny.h5 is in a different format, thus will not work with the current code
"""

# DATA = 'C:\\Users\\rotem\PycharmProjects\ML4Jets\Data\events_anomalydetection_tiny.h5'
DATA = 'C:\\Users\\rotem\PycharmProjects\ML4Jets\ML4Jets-HUJI\jets_data_100000.h5'
# DATA = 'C:\\Users\\rotem\PycharmProjects\ML4Jets\Data\events_anomalydetection.h5'
# DATA = '/Users/rotemmayo/Documents/PyCharm/ML4Jets-HUJI/jets_data_tiny.h5'


MODEL = 3
CHUNK_SIZE = 2**15
N_EVENTS = 2**13
IS_JETS = True
TEST_PERCENT = 0.2
TOTAL_SIZE = 1100000
DATA_SET_NAME = 'dataset_1'


"""
Hyper-parameters
"""
EPOCHS = 10
BATCH_SIZE = 16
DROPOUT = 0.1
LEARNING_RATE = 10**-1


"""
Create a callback that saves the model's weights
"""
NEW_MODEL = True
CP_PATH = "Model_0." + str(MODEL)+"/cp.ckpt"
CP_DIR = os.path.dirname(CP_PATH)
CP_CALLBACK = tf.keras.callbacks.ModelCheckpoint(filepath=CP_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)



"""
Option 1: Load everything into memory
"""


def read_h5(num_events=N_EVENTS):
    df = pd.read_hdf(DATA, stop=num_events)
    if IS_JETS:
        df.fillna(0, inplace=True)
    tf.keras.utils.normalize(df, axis=-1, order=2)
    print("Memory in GB:", sum(df.memory_usage(deep=True)) / (1024 ** 3))
    print(df.shape)
    return df


"""
Option 3: Use generator to loop over the whole file
Should be compatible with jets as well
"""


def generator(filename, chunksize=512, total_size=1100000):
    i = 2
    while True:
        yield create_dataset(pd.read_hdf(filename, start=i * chunksize, stop=(i + 1) * chunksize))
        i += 1
        if (i + 1) * chunksize > total_size:
            i = 2


"""
Creates a tensorflow data set from a pandas dataframe, assuming the target value is in the last column.
"""

"""
def create_dataset_jets(df):
    target = df.iloc[:, 0].astype(int)
    df = df.iloc[:, 1:]
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    shuffled_dataset = dataset.shuffle(len(df)).batch(BATCH_SIZE)
    # for feat, targ in shuffled_dataset.take(5):
    #     print('Features: {}, Target: {}'.format(feat, targ))
    return shuffled_dataset
"""

"""
Creates a tensorflow data set from a pandas dataframe, assuming the target value is in the last column.
"""


def create_dataset(df):
    if IS_JETS:
        target = df.iloc[:, 0].astype(int)
        df = df.iloc[:, 1:]
    else:
        target = df.iloc[:, -1].astype(int)
        df = df.iloc[:, :-1]
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    shuffled_dataset = dataset.shuffle(len(df)).batch(BATCH_SIZE)
    # for feat, targ in shuffled_dataset.take(5):
    #    print('Features: {}, Target: {}'.format(feat, targ))
    return shuffled_dataset


"""
For detector events dropout + batchnorm
"""
def model_one():
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
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


"""
For detector level events only batch_norm
"""


def model_two():
    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


"""
For jets, initial model dropout + batch_norm
"""


def model_three():
    model = tf.keras.Sequential([
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


"""
TODO: Add proper initialization
"""


def learn(df):
    msk = np.random.rand(len(df)) > TEST_PERCENT
    train = df[msk]
    test = df[~msk]
    test_dataset = create_dataset(test)
    train_dataset = create_dataset(train)
    model = load_model()
    model.fit(train_dataset, epochs=EPOCHS, verbose=2, validation_data=test_dataset, callbacks=[CP_CALLBACK])
    model.evaluate(test_dataset)


"""
Loads the correct model
"""


def load_model():
    if (MODEL == 1):
        model = model_one()
    elif (MODEL == 2):
        model = model_two()
    elif (MODEL == 3):
        model = model_three()
    if not NEW_MODEL:
        model.load_weights(CP_PATH).expect_partial()
    return model



"""
Learns using the generator
"""


def learn_generator():
    model = load_model()
    g1 = generator(DATA, chunksize=CHUNK_SIZE)
    g2 = generator(DATA, chunksize=CHUNK_SIZE)
    validation_set = g2.__next__()
    for j in range(EPOCHS):
        for i in range(TOTAL_SIZE//CHUNK_SIZE):
            train_dataset = g1.__next__()
            model.fit(train_dataset, epochs=1, verbose=2, validation_data=validation_set, callbacks=[CP_CALLBACK])
        print("EPOCH" + str(j))
    test_dataset = g2.__next__()
    model.evaluate(test_dataset)



"""
Evaluates the model
"""


def test_model():
    model = load_model()
    g2 = generator(DATA, chunksize=CHUNK_SIZE)
    g2.__next__()
    test_dataset = g2.__next__()
    model.evaluate(test_dataset)


def main():
    df = read_h5(100000)
    learn(df)
    # learn_generator()
    # test_model()


main()