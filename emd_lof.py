import random
import time

import numpy as np
import energyflow as ef
from get_jets import get_events_from_training_data
from tqdm import tqdm
from xlwt import Workbook
import os
import itertools
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from functools import wraps
import h5py

from pyjet import cluster, DTYPE_PTEPM
from tqdm import tqdm
from event import Event

TRAINING_DATA_FILE_PATH = './Data/events_anomalydetection_tiny.h5'
BLACK_BOX_FILE_PATH = './Data/BlackBox/events_LHCO2020_BlackBox%d.h5'
PARSED_DATA_FILE_PATH = './Data/Parsed_Data/%s.h5'

MAX_EVENT_PARTICLE_NUMBER_DIV_3 = 2100 / 3


def memoize(function):
    memo = {}
    counter =[0]
    @wraps(function)
    def wrapper(ev0, ev1):
        counter[0] += 1
        if counter[0] % 1000 == 0:
            print counter[0], "&",
        k = (ev0[0], ev1[1])
        if k in memo:
            return memo[k]
        if (ev1[0], ev0[1]) in memo:
            return memo[(ev1[0], ev0[1])]
        rv = function(ev0, ev1)
        memo[k] = rv
        return rv
    return wrapper


@memoize
def get_emdval(ev0, ev1):
    """

    :param ev0: [[pt, eta, phi], [pt, eta, phi], [pt, eta, phi]...]
    :param ev1: [[pt, eta, phi], [pt, eta, phi], [pt, eta, phi]...]
    :return:
    """
    d = ef.emd.emd(remove_zeros(ev0), remove_zeros(ev1), R=1)
    return d


def remove_zeros(ev):
    no_zeros = np.trim_zeros(ev)
    return no_zeros.reshape((int(len(no_zeros) / 3), 3))

    # nz_ev = []
    # for row in ev:
    #     rs_row = row.reshape(2100)
    #     no_zeros = np.trim_zeros(rs_row[:2100])
    #     nz_ev.append(no_zeros.reshape((int(len(no_zeros) / 3), 3)))
    # return nz_ev


def get_events():
    number_of_events = 1000
    fnew = pd.read_hdf(TRAINING_DATA_FILE_PATH, stop=number_of_events)
    npy_arr = fnew.to_numpy()
    is_signal_array = npy_arr[:,2100]
    # events = []
    # for row in npy_arr:
    #     no_zeros = np.trim_zeros(row[:2100])
    #     events.append(no_zeros.reshape((int(len(no_zeros) / 3), 3)))

    events = npy_arr[:,:2100]
    # events = npy_arr[:,:2100].reshape((len(npy_arr), 700, 3))
    # no_zeros_events = [[p for p in e if p[0] != 0] for e in events]

    is_signal = lambda i: is_signal_array[i]

    return events, is_signal


def lof():
    print 'Reading Events... ', time.time()

    data_set, is_signal = get_events()
    count_original_signal = len([i for i in range(len(data_set)) if is_signal(i)])
    print count_original_signal

    # print 'Scaling Data... ', time.time()
    # sc = StandardScaler()
    # scaled_data_set = sc.fit_transform(data_set)

    # perdictor = LocalOutlierFactor(metric=get_emdval, n_jobs=10, n_neighbors=200, contamination=0.09)
    # perdictor = LocalOutlierFactor(metric=get_emdval, n_jobs=20, n_neighbors=200, contamination=0.4)
    # perdictor = LocalOutlierFactor(n_jobs=20, n_neighbors=200, contamination=0.4)
    #
    # print 'Fitting...', time.time()
    # y_pred = perdictor.fit_predict(data_set)
    #
    # print 'Results...', time.time()
    # pred_signal = [i for i in range(len(y_pred)) if y_pred[i] != 1]
    # count_real_signal = len([i for i in pred_signal if is_signal(i)])
    #
    #
    # print("total tested " + str(len(y_pred)))
    # print("total original signal " + str(count_original_signal))
    # print("found signal " + str(len(pred_signal)))
    # print("real  signal " + str(count_real_signal))
    # print("ratio signal " + str(1.0*count_real_signal / len(pred_signal)))
    # print("recall: " + str(count_real_signal/count_original_signal))


def main():
    lof()


main()


# for tiny (1000 events):
# comparisons: 1,068,000
# start: 1632503438.23
# end: 1632506147.41
# total tested 1000
# found signal 90
# real  signal 6
# ratio signal 0

# total tested 1000
# total original signal 93
# found signal 400
# real  signal 46
# ratio signal 0.115

