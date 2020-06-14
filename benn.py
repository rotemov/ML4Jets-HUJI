from event_file_parser import EventFileParser
from get_jets import get_jets_from_training_data
import numpy as np

# NUMBER_OF_EVENTS = 100000
# signal_events, background_events, _ = get_jets_from_training_data(NUMBER_OF_EVENTS)
# all_events = signal_events + background_events


TRAINING_DATA_FILE_PATH = './Data/events_anomalydetection.h5'

parser = EventFileParser(TRAINING_DATA_FILE_PATH, '', chunk_size=10, total_size=100)
parser.parse()

# a numpy 2d array
all_events_array = np.array(parser.all_events['background'])
file_name = './Data/Parsed_Data/all'

np.save(file_name, all_events_array)


