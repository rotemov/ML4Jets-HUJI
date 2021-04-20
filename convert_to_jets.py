from __future__ import print_function, division

from datetime import datetime
import numpy as np
from pyjet import cluster,DTYPE_PTEPM
import pandas as pd
import h5py

NUMBER_OF_EVENTS = 100000
FILE_NAME = 'jets_data_100000.h5'
DATA_SET_NAME = 'dataset_1'

"""
TODO: Zero padding
TODO: Parse larger file
"""



def cluster_to_jets(events_combined):
    alljets = {}
    for mytype in ['background', 'signal']:
        alljets[mytype] = []
        #for i in range(np.shape(events_combined)[1]):
        for i in range(NUMBER_OF_EVENTS):
            if (i % (NUMBER_OF_EVENTS/10) == 0):
                print(mytype, i)
                pass
            issignal = events_combined[i][2100]
            if (mytype == 'background' and issignal):
                continue
            elif (mytype == 'signal' and issignal == 0):
                continue
            pseudojets_input = np.zeros(len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
            for j in range(700):
                if (events_combined[i][j * 3] > 0):
                    pseudojets_input[j]['pT'] = events_combined[i][j * 3]
                    pseudojets_input[j]['eta'] = events_combined[i][j * 3 + 1]
                    pseudojets_input[j]['phi'] = events_combined[i][j * 3 + 2]
                    pass
                pass
            sequence = cluster(pseudojets_input, R=1.0, p=-1)
            jets = sequence.inclusive_jets(ptmin=20)
            alljets[mytype] += [jets]
            pass
    return alljets



def convert_all(alljets):
    signal_jets = alljets['signal']
    background_jets = alljets['background']
    new_alljets = convert_to_x_y_z(signal_jets, 1) + convert_to_x_y_z(background_jets, 0)
    return new_alljets


def convert_to_x_y_z(jets, is_signal):
    events = []
    for event in jets:
        event_cart = [is_signal]
        for jet in event:
            jet_cart = [jet.e, jet.px, jet.py,jet.pz]
            event_cart += jet_cart
        events.append(event_cart)
    return events


def main():
    print(datetime.now())
    fnew = pd.read_hdf("/Users/rotemmayo/Documents/PyCharm/Data/events_anomalydetection.h5")
    events_combined = fnew.T
    print(np.shape(events_combined))
    res = convert_all(cluster_to_jets(events_combined))
    df = pd.DataFrame.from_records(res)
    print(df.shape)

    df.to_hdf(FILE_NAME, DATA_SET_NAME, mode='w', format='table')
    hf = h5py.File(FILE_NAME, 'a')
    hf.close()
    print(datetime.now())


if __name__ == "__main__":
    main()




