from event_file_parser import EventFileParser
from get_jets import get_jets_from_training_data
import numpy as np

# NUMBER_OF_EVENTS = 100000
# signal_events, background_events, _ = get_jets_from_training_data(NUMBER_OF_EVENTS)
# all_events = signal_events + background_events

"""
File the data was created from.
"""

TRAINING_DATA_FILE_PATH = './Data/events_anomalydetection.h5'

parser = EventFileParser(TRAINING_DATA_FILE_PATH, '')
parser.parse()

# a numpy 2d array
bg_events_array = np.array(parser.all_events['background'])
file_name = './Data/Parsed_Data/bg'
np.save(file_name, bg_events_array)
sig_events_array = np.array(parser.all_events['signal'])
file_name = './Data/Parsed_Data/sig'
np.save(file_name, sig_events_array)

print("Done")


