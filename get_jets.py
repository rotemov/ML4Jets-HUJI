import pandas as pd
import numpy as np
from pyjet import cluster, DTYPE_PTEPM
from tqdm import tqdm
import h5py

BLACK_BOX_FILE_PATH = './Data/BlackBox/events_LHCO2020_BlackBox%d.h5'
PARSED_DATA_FILE_PATH = './Data/Parsed_Data/%s.h5'

MAX_EVENT_PARTICLE_NUMBER = 700


def get_jets_from_black_box(box_number, number_of_events=None, R=1.0):
    """
    For each event in the file, get the jets relating to the event.
    Clustering using pyjet.
    For each event, jets are ordered in descending order of pt

    :param box_number: The number of the black box to read
    :param number_of_events: The number of events to read, default to all of them
    :param R: The jet radius to cluster bt, defaults to 1.0
    :return: list of lists of all jets relating to one event
    """
    fnew = pd.read_hdf(BLACK_BOX_FILE_PATH % box_number, stop=number_of_events)
    events_combined = fnew.T

    all_jets = []
    for i in tqdm(range(np.shape(events_combined)[1])):
        pseudojets_input = np.zeros(len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM)

        for j in range(MAX_EVENT_PARTICLE_NUMBER):
            if events_combined[i][j * 3] > 0:
                pseudojets_input[j]['pT'] = events_combined[i][j * 3]
                pseudojets_input[j]['eta'] = events_combined[i][j * 3 + 1]
                pseudojets_input[j]['phi'] = events_combined[i][j * 3 + 2]

        sequence = cluster(pseudojets_input, R=R, p=-1)
        jets = sequence.inclusive_jets(ptmin=20)
        all_jets += [jets]

    return all_jets


def save_jets_to_file(box_number, new_file_name, data_set_name='dataset_1', number_of_events=None, R=1.0):
    """
    Convert events from a black box file to an h5 file containing all jets in cartesian coordinates

    :param box_number: The number of the black box to read
    :param new_file_name: The name of the new file to create
    :param data_set_name: The name of the data set to create
    :param number_of_events: The number of events to read, default to all of them
    :param R: The jet radius to cluster bt, defaults to 1.0
    """
    all_jets = get_jets_from_black_box(box_number, number_of_events, R)
    cartesian_all_jets = _cartesian(all_jets)
    df = pd.DataFrame.from_records(cartesian_all_jets)
    new_file_path = PARSED_DATA_FILE_PATH % new_file_name
    df.to_hdf(new_file_path, data_set_name, mode='w', format='table')
    hf = h5py.File(new_file_path, 'a')
    hf.close()


def _cartesian(all_jets):
    """
    Convert to cartesian coordinates

    :param all_jets: list of lists of all jets relating to one event
    :return: all_jets, each jet is a list of cartesian coordinates [e, px, py, pz]
    """
    cartesian_all_jets = []
    for jets in all_jets:
        cartesian_jets = []
        for jet in jets:
            cartesian_jets += [jet.e, jet.px, jet.py, jet.pz]
        cartesian_all_jets += cartesian_jets
    return cartesian_all_jets

