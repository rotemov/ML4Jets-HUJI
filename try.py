from get_jets import  get_jets_from_training_data
from plotter import Plotter

R = 1.0
NUMBER_OF_EVENTS = 1000

signal_events, background_events, _ = get_jets_from_training_data(NUMBER_OF_EVENTS)

plotter = Plotter(signal_events, background_events)
plotter.plot_histogram(lambda e: e.mjj, 'mjj', '#events')
plotter.plot_histogram(lambda e: e.m_tot, 'tot', '#events')
plotter.plot_scatter(lambda e: max(e.m1, e.m2), lambda e: e.m1_minus_m2, 'm1', '|m1-m2|')

for event in signal_events:
    print(event.nsubjettiness)
