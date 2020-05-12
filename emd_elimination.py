import random
import numpy as np
import energyflow as ef
from get_jets import get_events_from_training_data


def get_emdval(ev0, ev1):
    return ef.emd.emd(ev0, ev1, R=0.4)


# Read Events
NUMBER_OF_EVENTS = 1000
VERBOSE = False
events_dat = get_events_from_training_data(NUMBER_OF_EVENTS)

# Choose a background event randomly as a start filter event
background_events = [e for e in events_dat if not e[0][3]]
random_index = random.randrange(len(background_events))
filter_event = background_events[random_index]
print('--- Start Event Index: %d ---' % random_index)

start_signal_num = len([e for e in events_dat if e[0][3]])
start_events_num = len(events_dat)

min_emd = 0
EMD_CUT_OFF = 5000
while min_emd < EMD_CUT_OFF and len(events_dat) > 0:
    events_dat.remove(filter_event)
    min_emd = 100000000
    min_event = None

    for e in events_dat:
        emdval = get_emdval(np.array([l[:3] for l in filter_event]), np.array([l[:3] for l in e]))
        if 0 < emdval < min_emd:
            min_emd = emdval
            min_event = e

    if VERBOSE:
        print 'Filter event: is signal: %d, emd from last: %f' % (min_event[0][3], min_emd)
    filter_event = min_event

end_signal_num = len([e for e in events_dat if e[0][3]])
end_events_num = len(events_dat)

print 'num events: start: %d, end: %d' % (start_events_num, end_events_num)
print 'sig events: start: %d, end: %d' % (start_signal_num, end_signal_num)
