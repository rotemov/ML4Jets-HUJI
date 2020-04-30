import nsubjettiness

from get_jets import get_jets_from_black_box
from jupyter_methods import plot_scatter, get_m1_sub_m2, get_m1, get_m2
from nsubjettines_fjcontrib import get_fjcontrib_nsubjettiness

R = 1.0
NUMBER_OF_EVENTS = 10

all_jets, events_combined = get_jets_from_black_box(1, NUMBER_OF_EVENTS, R=R)

# x = [max(get_m1(event_jets), get_m2(event_jets)) for event_jets in all_jets]
# y = [get_m1_sub_m2(event_jets) for event_jets in all_jets]
# plot_scatter(x, y, 'm1', '|m1-m2|')


# print('### FJ CONTRIB ###')
# get_fjcontrib_nsubjettiness(events_combined[0])

# print('### GEORGE ###')
first_event = all_jets[0]
for jet in first_event:
    print(jet)
    print('tau1: %f, tau2: %f, tau3: %f, tau4: %f\n' % tuple(nsubjettiness.get_nsubjettiness(jet, R=R)))
