from get_jets import get_jets_from_training_data
from plotter import Plotter

R = 1.0
NUMBER_OF_EVENTS = 5000

signal_events, background_events, _ = get_jets_from_training_data(NUMBER_OF_EVENTS)

plotter = Plotter(signal_events, background_events)
# plotter.plot_scatter(lambda e: max(e.m1, e.m2), lambda e: e.m1_minus_m2, 'm1', '|m1-m2|')


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
