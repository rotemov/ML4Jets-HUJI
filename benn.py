from event_file_parser import EventFileParser
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')
"""
File the data was created from.
"""

R_VALUES = [0.4, 1.0]

TRAINING_DATA_FILE_PATH = 'Data/events_anomalydetection.h5'

SIG_FILE_FORMAT = "{}sig_{}.npy"
BG_FILE_FORMAT = "{}bg_{}.npy"
DATA_PATH = "Data/Parsed_Data/"

CSV_FILE_PATH = "{}obs_parton_R{}.csv".format(DATA_PATH, "{}")

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
EXPERIMENT_MJJ_TRANSLATION = [0, 0, 0, 0, 10 ** 2, 10 ** 3, 5 * 10 ** 3, 10 ** 4, 0, 0, 0]


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
    R = 1.0
    for R in [1.0, 0.4, 0.7]:
        p = EventFileParser(TRAINING_DATA_FILE_PATH, CSV_FILE_PATH.format(R), R=R)
        p.parse()
    # create_full_data(DATA_PATH)  # need to shuffle jets and partons for supervised training
    """
    for i in range(len(EXPERIMENT_NAMES)):
        for R in R_VALUES:
            full_data_name = "R" + str(R)
            create_partial_data_set(full_data_name, DATA_PATH, EXPERIMENT_NAMES[i], EXPERIMENT_OBSERVABLES[i],
                                    OBS_DICT_ITERATION_DIFFERENT_R, EXPERIMENT_MJJ_TRANSLATION[i])
    """


def reorganize_data():
    num_events = 1100288
    num_chunks = 10787
    chunk_size = int(num_events / num_chunks)
    bg_only = pd.HDFStore('{}bg_data_truthbit_mjj_tau21.h5'.format(DATA_PATH), complib='zlib')
    combined = pd.HDFStore('{}combined_data_truthbit_mjj_tau21.h5'.format(DATA_PATH), complib='zlib')
    mjj_tau21_cols = [176, 185]
    mjj_tau21 = pd.read_csv(CSV_FILE_PATH.format(0.7), usecols=mjj_tau21_cols)
    n_bg = 0
    for i in tqdm(range(num_chunks)):
        start = chunk_size * i
        stop = chunk_size * (i+1)
        df = pd.read_hdf(TRAINING_DATA_FILE_PATH, start=start, stop=stop)
        df["mjj"] = mjj_tau21.values[start:stop, 0]
        df["tau21"] = mjj_tau21.values[start:stop, 1]
        mask = df['2100'] == 0
        if n_bg < 5*10**5:
            bg_only.append('data', df[mask])
            combined.append('data', df[~mask])
        else:
            combined.append('data', df)
    bg_only.close()
    combined.close()


def plot_1d_histograms(obs, sig_mask, prefix, xlabel):
    sig = obs[sig_mask]
    bg = obs[~sig_mask]
    # fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    n_bins = [10, 20, 30]
    for i, n in enumerate(n_bins):
        # combined = trim_outliers(np.hstack((sig[:, i], bg[:, i])), trim_percent)
        plt.figure()
        bins = np.histogram(obs, bins=n)[1]
        plt.hist(obs, bins=bins, label="combined", color="purple", log=True)
        plt.hist(bg, color="b", label="bg", log=True, bins=bins)
        plt.hist(sig, color="r", label="sig", log=True, bins=bins)
        plt.legend()
        plt.title("{}, N={}".format(prefix, n))
        plt.xlabel(xlabel)
        plt.ylabel("Num events")
        plt.savefig("{}histograms/1d_{}_nbins{}.png".format(DATA_PATH, prefix, n))
        plt.close()


def plot_2d_histograms(mjj, tau21, prefix):
    n_bins = [10, 20, 30]
    for n_mjj in n_bins:
        for n_tau21 in n_bins:
            plt.figure()
            plt.hist2d(mjj, tau21, bins=[n_mjj, n_tau21])
            plt.title("{}, Nmjj={}, Ntau21={}".format(prefix, n_mjj, n_tau21))
            plt.xlabel("$M_{jj}$")
            plt.ylabel("$$\\tau_{21}$")
            plt.savefig("{}histograms/2d_{}_nmjj{}_ntau21{}.png".format(DATA_PATH, prefix, n_mjj, n_tau21))
            plt.close()


if __name__ == "__main__":
    mjj_tau21_sig_cols = [176, 185, 189]
    mjj_tau21_sig = pd.read_csv(CSV_FILE_PATH.format(0.7), usecols=mjj_tau21_sig_cols).values
    sig_mask = mjj_tau21_sig[:, 2] == 1
    plot_1d_histograms(mjj_tau21_sig[:, 0], sig_mask, "mjj", "$M_{jj}[GeV]$")
    plot_1d_histograms(mjj_tau21_sig[:, 1], sig_mask, "tau21", "$\\tau_{21}$")
    sig = mjj_tau21_sig[sig_mask]
    bg = mjj_tau21_sig[~sig_mask]
    plot_2d_histograms(sig[:, 0], sig[:, 1], "sig")
    plot_2d_histograms(bg[:, 0], bg[:, 1], "bg")
    plot_2d_histograms(mjj_tau21_sig[:, 0], mjj_tau21_sig[:, 1], "combined")
    reorganize_data()
