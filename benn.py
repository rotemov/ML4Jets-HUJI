from event_file_parser import EventFileParser
import numpy as np
import os

"""
File the data was created from.
"""

R_VALUES = [0.4, 1.0]

TRAINING_DATA_FILE_PATH = 'Data/events_anomalydetection.h5'

SIG_FILE_FORMAT = "{}sig_{}.npy"
BG_FILE_FORMAT = "{}bg_{}.npy"
DATA_PATH = "Data/Parsed_Data/"
OBS_DICT_ITERATION_1 = {
    'mjj': 0, "nj": 1, "mtot": 2, "m1": 3, "m2": 4, "tau21_1": 5, "tau21_2": 6
}

OBS_DICT_ITERATION_DIFFERENT_R = {
    'mjj': 0, "nj": 1, "mtot": 2, "m1": 3, "m2": 4, "m1_minus_m2": 5, "lead_pt": 6,
    "ht": 7, "mht": 8, "tau21_1": 9, "tau21_2": 10
}

EXPERIMENT_NAMES = ["mjj", "anode", "salad", "all", "all_mjj-translation_100", "all_mjj-translation_1000",
                    "all_mjj-translation_5000", "all_mjj-translation_10000", "mjj_m1", "mjj_m1_m1minusm2",
                    "mjj_m1minusm2"]
EXPERIMENT_OBSERVABLES = [
    ["mjj"],
    ["m1", "m1_minus_m2", "tau21_1", "tau21_2"],
    ["m1", "m2", "tau21_1", "tau21_2"],
    ['mjj', "nj", "mtot", "m1", "m2", "m1_minus_m2", "lead_pt", "ht", "mht", "tau21_1", "tau21_2"],
    ['mjj', "nj", "mtot", "m1", "m2", "m1_minus_m2", "lead_pt", "ht", "mht", "tau21_1", "tau21_2"],
    ['mjj', "nj", "mtot", "m1", "m2", "m1_minus_m2", "lead_pt", "ht", "mht", "tau21_1", "tau21_2"],
    ['mjj', "nj", "mtot", "m1", "m2", "m1_minus_m2", "lead_pt", "ht", "mht", "tau21_1", "tau21_2"],
    ['mjj', "nj", "mtot", "m1", "m2", "m1_minus_m2", "lead_pt", "ht", "mht", "tau21_1", "tau21_2"],
    ["mjj", "m1"],
    ["mjj", "m1", "m1_minus_m2"],
    ["mjj", "m1_minus_m2"]
]
EXPERIMENT_MJJ_TRANSLATION = [0, 0, 0, 0, 10**2, 10**3, 5*10**3, 10**4, 0, 0, 0]


def create_full_data(path):
    for R in R_VALUES:
        print("Starting parse R{}".format(R))
        parser = EventFileParser(TRAINING_DATA_FILE_PATH, '', R=R)
        parser.parse()
        print("Parsed R{}".format(R))
        # data is a numpy 2d array
        bg_events_array = np.array(parser.all_events['background'])
        file_name = '{}bg_partons_R{}'.format(path, R)
        np.save(file_name, bg_events_array)
        sig_events_array = np.array(parser.all_events['signal'])
        file_name = '{}sig_partons_R{}'.format(path, R)
        np.save(file_name, sig_events_array)
        print("Data sets creates R{}".format(R))


def create_partial_data_set(full_data_name, data_path, experiment_name, obs_list, obs_dict, sig_mjj_translation=0):
    bg = np.load(BG_FILE_FORMAT.format(data_path, full_data_name))
    sig = np.load(SIG_FILE_FORMAT.format(data_path, full_data_name))
    idx = [obs_dict[obs_name] for obs_name in obs_list]
    new_sig = sig[:, idx]
    new_bg = bg[:, idx]
    if "mjj" in obs_list and "mjj" in obs_dict.keys():
        new_sig[:, 0] += sig_mjj_translation
    name = full_data_name + "_" + experiment_name
    np.save(BG_FILE_FORMAT.format(data_path, name), new_bg)
    np.save(SIG_FILE_FORMAT.format(data_path, name), new_sig)


def main():
    create_full_data(DATA_PATH)  # need to shuffle jets and partons for supervised training
    """
    for i in range(len(EXPERIMENT_NAMES)):
        for R in R_VALUES:
            full_data_name = "R" + str(R)
            create_partial_data_set(full_data_name, DATA_PATH, EXPERIMENT_NAMES[i], EXPERIMENT_OBSERVABLES[i],
                                    OBS_DICT_ITERATION_DIFFERENT_R, EXPERIMENT_MJJ_TRANSLATION[i])
    """


if __name__ == "__main__":
    main()

