import random
import numpy as np
import energyflow as ef
from get_jets import get_events_from_training_data
from tqdm import tqdm
from xlwt import Workbook
import os
import itertools
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


EMD_CUT_OFF = 5000


def get_emdval(ev0, ev1):
    """

    :param ev0: [[pt, eta, phi], [pt, eta, phi], [pt, eta, phi]...]
    :param ev1: [[pt, eta, phi], [pt, eta, phi], [pt, eta, phi]...]
    :return:
    """
    return ef.emd.emd(ev0, ev1, R=0.4)


def create_xls(events_list, name):
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    for i in tqdm(range(len(events_list))):
        f_event = events_list[i]
        sheet1.write(i + 1, 0, 'Sig' if f_event[0][3] else 'Bkg')
        sheet1.write(0, i + 1, 'Sig' if f_event[0][3] else 'Bkg')
        for j in range(len(events_list)):
            s_event = events_list[j]
            emd = get_emdval(np.array([l[:3] for l in f_event]), np.array([l[:3] for l in s_event]))
            sheet1.write(i + 1, j + 1, str(int(emd)))
    wb.save(name + '.xls')


def create_csv(events_list, name):
    with open(name, 'w') as f:
        for e in events:
            f.write(',Sig' if e[0][3] else ',Bkg')
        f.write('\n')

        for i in tqdm(range(len(events_list))):
            f_event = events_list[i]
            line = 'Sig' if f_event[0][3] else 'Bkg'
            for j in range(len(events_list[:i+1])):
                s_event = events_list[j]
                emd = get_emdval(np.array([l[:3] for l in f_event]), np.array([l[:3] for l in s_event]))
                line += ',' + str(int(emd))
            f.write(line + '\n')
            f.flush()


def filter_out_events(background_events_list, events_list):
    random_index = random.randrange(len(background_events_list))
    filter_event = background_events_list[random_index]
    print('--- Start Event Index: %d ---' % random_index)
    start_signal_num = len([e for e in events_list if e[0][3]])
    start_events_num = len(events_list)
    min_emd = 0
    pbar = tqdm(total=NUMBER_OF_EVENTS)
    while min_emd < EMD_CUT_OFF and len(events_list) > 0:
        events_list.remove(filter_event)
        min_emd = 100000000
        min_event = None

        for e in events_list:
            emdval = get_emdval(np.array([l[:3] for l in filter_event]), np.array([l[:3] for l in e]))
            if 0 < emdval < min_emd:
                min_emd = emdval
                min_event = e

        if VERBOSE:
            print 'Filter event: is signal: %d, emd from last: %f' % (min_event[0][3], min_emd)
        filter_event = min_event
        pbar.update(1)
    pbar.close()
    end_signal_num = len([e for e in events_list if e[0][3]])
    end_events_num = len(events_list)
    print 'num events: start: %d, end: %d' % (start_events_num, end_events_num)
    print 'sig events: start: %d, end: %d' % (start_signal_num, end_signal_num)


def bins(events):
    e1 = events[0]
    e2 = events[1]
    events.remove(e1)
    events.remove(e2)
    bin1 = [e1]
    maxd1 = 0
    bin2 = [e2]
    maxd2 = 0

    for event in events:
        d1 = get_emdval(np.array([l[:3] for l in e1]), np.array([l[:3] for l in event]))
        d2 = get_emdval(np.array([l[:3] for l in e2]), np.array([l[:3] for l in event]))
        if d1 < d2:
            bin1 += [event]
            if d1 > maxd1:
                maxd1 = d1
        else:
            bin2 += [event]
            if d2 > maxd2:
                maxd2 = d2

    ret_bins = []
    if maxd1 > EMD_CUT_OFF:
        bins1 = bins(bin1)
        ret_bins = ret_bins + bins1
    else:
        ret_bins = ret_bins + [bin1]

    if maxd2 > EMD_CUT_OFF:
        bins2 = bins(bin2)
        ret_bins = ret_bins + bins2
    else:
        ret_bins = ret_bins + [bin2]

    return ret_bins


def get_bin_mean_dist(b):
    r = []
    for pair in itertools.product(b, repeat=2):
        (e1, e2) = pair
        d = get_emdval(np.array([l[:3] for l in e1]), np.array([l[:3] for l in e2]))
        r += [d]
    return np.mean(r)


# Read Events
NUMBER_OF_EVENTS = 1000
VERBOSE = False
print 'Read Events'
events_dat = get_events_from_training_data(NUMBER_OF_EVENTS)

# Choose a background event randomly as a start filter event
background_events = [e for e in events_dat if not e[0][3]]
signal_events = [e for e in events_dat if e[0][3]]
events = signal_events + background_events

print 'saving to: ' + os.path.abspath('emd_test.csv')
create_csv(events, 'emd_test.csv')
# filter_out_events(background_events, events_dat)

# b = bins(events_dat)
# b.sort(key=len)
# print 'num of bins: %d' % (len(b))
# for bin in b:
#     s = len([e for e in bin if e[0][3]])
#     print '     bin size: %d, bins mean dist: %d, signal: %d' % (len(bin), get_bin_mean_dist(bin), s)
#
# os.system('say "Finished"')
