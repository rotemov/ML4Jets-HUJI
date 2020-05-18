from get_jets import get_jets_from_training_data
from plotter import Plotter
import numpy as np
from event_file_parser import EventFileParser
from datetime import datetime
from pyjet import PseudoJet

DATA_PATH = "Data/events_anomalydetection_tiny.h5"
JSON_NAME = "test_json"

print(datetime.now())
p = EventFileParser(DATA_PATH, JSON_NAME, total_size=10)
p.parse()
print(datetime.now())
frozen = p.events_to_json()
print(datetime.now())
thawed = EventFileParser.events_from_json(frozen)
print(datetime.now())
print(thawed)
e = thawed['background'][1]
e1 = p.all_events['background'][1]
j = e.jets
j1 = e1.jets
print(e.index)
print(e.box)
print(e.nj, e1.nj)
print(e.mjj, e1.mjj)
print(j)
print(j[0].px, j1[0].px)



"""
R = 1.0
NUMBER_OF_EVENTS = 20

signal_events, background_events, events_combined = get_jets_from_training_data(NUMBER_OF_EVENTS)

hadrondata = events_combined.iloc[:,1:3]
j = 0
events = []
for e in hadrondata.iteritems():
    events += [[]]
    for i in range(0, 700):
        events[-1] += [[e[1][3*i], e[1][3*i+1], e[1][3*i+2]]]
    j += 1
print(len(events[0]))



plotter = Plotter(signal_events, background_events)
# plotter.plot_scatter(lambda e: max(e.m1, e.m2), lambda e: e.m1_minus_m2, 'm1', '|m1-m2|')
"""

"""
print('1')
plotter.plot_histogram(lambda e: e.all_jet_mass, 'jet mass', '#events')
print('2')
plotter.plot_histogram(lambda e: e.all_jet_pt, 'jet pt', '#events')
print('3')
plotter.plot_histogram(lambda e: e.nj, 'nj', '#events')
print('4')
plotter.plot_histogram(lambda e: e.mjj, 'mjj', '#events')
print('5')
plotter.plot_histogram(lambda e: e.mjj_all_pairs, 'mjj_all_pairs', '#events')
print('6')
plotter.plot_histogram(lambda e: e.m_tot, 'mtot', '#events')


# for event in signal_events:
#     print(event.nsubjettiness)
"""
