import itertools
import matplotlib.pyplot as plt
import numpy as np


def get_mtot(event_jets):
    all_px = sum([j.px ** 2 for j in event_jets])
    all_py = sum([j.py ** 2 for j in event_jets])
    all_pz = sum([j.pz ** 2 for j in event_jets])
    all_e = sum([j.e for j in event_jets])

    if all_e ** 2 - all_px - all_py - all_pz >= 0:
        return (all_e ** 2 - all_px - all_py - all_pz) ** 0.5
    else:
        raise Exception('Bad MTot: all_e=%d, all_px=%d, all_py=%d, all_pz=%d'.format(all_e, all_px, all_py, all_pz))


def get_mjj(event_jets):
    """
    The 2 first jets are the leading jets

    :param event_jets:
    :return: The mjj for the 2 leading jets
    """
    e = event_jets[0].e + event_jets[1].e
    px = event_jets[0].px + event_jets[1].px
    py = event_jets[0].py + event_jets[1].py
    pz = event_jets[0].pz + event_jets[1].pz
    return (e ** 2 - px ** 2 - py ** 2 - pz ** 2) ** 0.5


def get_mjj_all_pairs(event_jets):
    mjj_all_pairs = []
    for pair in itertools.product(event_jets, repeat=2):
        (jo, jt) = pair
        e = jo.e + jt.e
        px = jo.px + jt.px
        py = jo.py + jt.py
        pz = jo.pz + jt.pz
        if (e ** 2 - px ** 2 - py ** 2 - pz ** 2) >= 0:
            mjj_all_pairs += [(e ** 2 - px ** 2 - py ** 2 - pz ** 2) ** 0.5]
        else:
            raise Exception('Bad Mjj: e=%d, px=%d, py=%d, pz=%d'.format(e, px, py, pz))
    return mjj_all_pairs


def get_lead_pt(event_jets):
    return event_jets[0].pt


def get_nj(event_jets):
    return len(event_jets)


def get_mht(event_jets, pt_cutoff=30, eta_cutoff=5):
    all_px = np.array([jet.px for jet in event_jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)])
    all_py = np.array([jet.py for jet in event_jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)])
    return sum(np.square(all_px) + np.square(all_py)) ** 0.5


def get_ht(event_jets, pt_cutoff=30, eta_cutoff=2.5):
    all_px = np.array([jet.px for jet in event_jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)])
    all_py = np.array([jet.py for jet in event_jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)])
    return sum(np.square(all_px) + np.square(all_py)) ** 0.5


def get_meff(event_jets):
    all_px = np.array([jet.px for jet in event_jets])
    all_py = np.array([jet.py for jet in event_jets])
    return sum(jet.pt for jet in event_jets) + (sum(np.square(all_px) + np.square(all_py)))**0.5


def plot_histogram(data, x_label, y_label, color='b'):
    plt.figure()
    plt.hist(data, bins=50, facecolor=color, alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_scatter(x, y, x_label, y_label):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def get_m1(event_jets):
    jet1 = event_jets[0]
    return (jet1.e ** 2 - jet1.px ** 2 - jet1.py ** 2 - jet1.pz ** 2) ** 0.5


def get_m2(event_jets):
    jet2 = event_jets[1]
    return (jet2.e ** 2 - jet2.px ** 2 - jet2.py ** 2 - jet2.pz ** 2) ** 0.5


def get_m1_sub_m2(event_jets):
    return abs(get_m1(event_jets) - get_m2(event_jets))
