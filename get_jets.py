import pandas as pd
import numpy as np
import h5py

from pyjet import cluster, DTYPE_PTEPM
from tqdm import tqdm
from event import Event

TRAINING_DATA_FILE_PATH = './Data/events_anomalydetection.h5'
BLACK_BOX_FILE_PATH = './Data/BlackBox/events_LHCO2020_BlackBox%d.h5'
PARSED_DATA_FILE_PATH = './Data/Parsed_Data/%s.h5'

MAX_EVENT_PARTICLE_NUMBER_DIV_3 = 2100 / 3


def get_jets_from_black_box(box_number, number_of_events=None, R=1.0, eta_cut=2.5):
    """
    Parse the black box file (according to the given number) into Event objects.
    Each event contains all jets relating to it (within the given eta cut).
    Clustering is done using pyjet.
    For each event, jets are ordered in descending order of pt.

    :param box_number: The number of the black box to read, used in the data file name
    :param number_of_events: The number of events to read, default to all of them
    :param R: The jet radius to cluster by, defaults to 1.0
    :param eta_cut: Max eta to allow. Jets not within this cut will be silently dropped

    :return: list Events, and events_combined - the raw data from the file
    """
    fnew = pd.read_hdf(BLACK_BOX_FILE_PATH % box_number, stop=number_of_events)
    events_combined = fnew.T

    events = []
    for i in tqdm(range(np.shape(events_combined)[1])):
        jets = _cluster_event_column(events_combined[i], R, eta_cut)
        if jets:
            events.append(Event(jets))

    return events, events_combined


def get_jets_from_training_data(number_of_events=None, R=1.0, eta_cut=2.5, file=TRAINING_DATA_FILE_PATH):
    """
    Parse the file grouped into signal and background into Event objects.
    Each event contains all jets relating to it (within the given eta cut).
    Clustering is done using pyjet.
    For each event, jets are ordered in descending order of pt.

    :param number_of_events: The number of events to read, default to all of them
    :param R: The jet radius to cluster by, defaults to 1.0
    :param eta_cut: Max eta to allow. Jets not within this cut will be silently dropped

    :return: list of signal Events, list of background Events,
             and events_combined - the raw data from the file
    """
    fnew = pd.read_hdf(file, stop=number_of_events)
    events_combined = fnew.T

    signal_events = []
    background_events = []
    for i in tqdm(range(np.shape(events_combined)[1])):
        jets = _cluster_event_column(events_combined[i], R, eta_cut)

        if jets:
            if int(events_combined[i][2100]) == 1:
                signal_events.append(Event(jets))
            else:
                background_events.append(Event(jets))

    return signal_events, background_events, events_combined


def get_events_from_training_data(number_of_events):
    """
    Parse the file grouped into signal and background

    :param number_of_events: The number of events to read, default to all of them

    :return: list of list, each list represents an event and contains a list of lists containing
             [pt, eta, phi, is_signal]
    todo: find a better way to return is signal
    """
    fnew = pd.read_hdf(TRAINING_DATA_FILE_PATH, stop=number_of_events)
    events_combined = fnew.T

    events = []
    for i in tqdm(range(np.shape(events_combined)[1])):
        event_column = events_combined[i]
        is_signal = int(event_column[2100]) == 1
        event_data = []
        for j in range(MAX_EVENT_PARTICLE_NUMBER_DIV_3):
            if event_column[j * 3] > 0:
                pt = event_column[j * 3]
                eta = event_column[j * 3 + 1]
                phi = event_column[j * 3 + 2]

                event_data += [[pt, eta, phi, is_signal]]
        events += [event_data]

    return events


def save_jets_to_file(box_number, new_file_name, data_set_name='dataset_1', number_of_events=None, R=1.0):
    """
    Convert events from a black box file to an h5 file containing all jets in cartesian coordinates

    :param box_number: The number of the black box to read
    :param new_file_name: The name of the new file to create
    :param data_set_name: The name of the data set to create
    :param number_of_events: The number of events to read, default to all of them
    :param R: The jet radius to cluster by, defaults to 1.0
    """
    events, _ = get_jets_from_black_box(box_number, number_of_events, R)
    cartesian_all_jets = _cartesian(events)
    df = pd.DataFrame.from_records(cartesian_all_jets)
    new_file_path = PARSED_DATA_FILE_PATH % new_file_name
    df.to_hdf(new_file_path, data_set_name, mode='w', format='table')
    hf = h5py.File(new_file_path, 'a')
    hf.close()


def _cartesian(events):
    """
    Convert to cartesian coordinates

    :param events: list of Event objects
    :return: all_jets, a list of lists of jets Each jet is a list of cartesian coordinates [e, px, py, pz]
    """
    cartesian_all_jets = []
    all_jets = [event.jets for event in events]
    for jets in all_jets:
        cartesian_jets = []
        for jet in jets:
            cartesian_jets += [jet.e, jet.px, jet.py, jet.pz]
        cartesian_all_jets += cartesian_jets
    return cartesian_all_jets


def _cluster_event_column(event_column, R, eta_cut):
    """
    Cluster a raw event from an h5 file into jets. Cluster using pyjet

    :param event_column: The raw event to cluster
    :param R: The jet radius to cluster by
    :param eta_cut: Max eta to allow. Jets not within this cut will be silently dropped

    :return: A list of the jets (within the cut) of the events.
             Jets are pyjet PseudoJet objects
    """
    pseudojets_input = np.zeros(len([x for x in event_column[::3] if x > 0]), dtype=DTYPE_PTEPM)
    for j in range(MAX_EVENT_PARTICLE_NUMBER_DIV_3):
        if event_column[j * 3] > 0:
            pseudojets_input[j]['pT'] = event_column[j * 3]
            pseudojets_input[j]['eta'] = event_column[j * 3 + 1]
            pseudojets_input[j]['phi'] = event_column[j * 3 + 2]
    sequence = cluster(pseudojets_input, R=R, p=-1)
    jets = sequence.inclusive_jets(ptmin=20)

    # cut jets not within |eta_cut|
    return [jet for jet in jets if abs(jet.eta) <= eta_cut]
