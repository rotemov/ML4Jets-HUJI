from event_file_parser import EventFileParser
import numpy as np
import os
import matplotlib.pyplot as plt


"""
File the data was created from.
"""

R_VALUE = 1.0

TRAINING_DATA_FILE_PATH = 'Data/events_anomalydetection.h5'


def create_full_data(path):
    R = 1.0
    print("Starting...")
    file_name = 'mayo_events_all_obs_big'
    parser = EventFileParser(path, file_name, R=R)
    parser.parse()
    print("Done!:)")


def plot_histogram(file_name, func, x_label, label_index):
    data = open(file_name, 'r').read()
    dataset = [[float(j) for j in i.split(',')] for i in data.split('\n') if len(i) > 0]
    sorted_dataset = sorted(dataset, key=lambda i: i[label_index])
    first_signal_index = len(sorted_dataset) - sum([i[label_index] for i in sorted_dataset])

    background = dataset[0:int(first_signal_index)]
    signal = dataset[int(first_signal_index):-1]

    signal_to_plot = [func(e) for e in signal]
    background_to_plot = [func(e) for e in background]

    plt.figure()
    plt.hist(background_to_plot, bins=50, facecolor='b', label='background')
    plt.hist(signal_to_plot, bins=50, facecolor='r', label='signal')
    plt.xlabel(x_label)
    plt.ylabel("Number of Events")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(x_label + '.png')


def main():
    # create_full_data(TRAINING_DATA_FILE_PATH)

    plot_histogram('./mayo_events_all_obs_big', lambda e: e[2]-e[3], "m1-m2 [GeV]", 4)
    plot_histogram('./mayo_events_all_obs_big', lambda e: e[2]+e[3], "m1+m2 [GeV]", 4)
    plot_histogram('./mayo_events_all_obs_big', lambda e: e[2], "m1 [GeV]", 4)
    plot_histogram('./mayo_events_all_obs_big', lambda e: e[3], "m2 [GeV]", 4)
    plot_histogram('./mayo_events_all_obs_big', lambda e: e[0], "mjj [GeV]", 4)
    plot_histogram('./mayo_events_all_obs_big', lambda e: e[1], "mtot [GeV]", 4)
    plot_histogram('./mayo_events', lambda e: e[0], "nj", 3)
    plot_histogram('./mayo_events', lambda e: e[1], "ht [GeV]", 3)
    plot_histogram('./mayo_events', lambda e: e[2], "mht [GeV]", 3)

    # R = 1.0
    # p = EventFileParser(TRAINING_DATA_FILE_PATH, CSV_FILE_PATH.format(R), R=R)
    # p.parse()
    # # create_full_data(DATA_PATH)  # need to shuffle jets and partons for supervised training
    """
    for i in range(len(EXPERIMENT_NAMES)):
        for R in R_VALUES:
            full_data_name = "R" + str(R)
            create_partial_data_set(full_data_name, DATA_PATH, EXPERIMENT_NAMES[i], EXPERIMENT_OBSERVABLES[i],
                                    OBS_DICT_ITERATION_DIFFERENT_R, EXPERIMENT_MJJ_TRANSLATION[i])
    """


if __name__ == "__main__":
    main()
