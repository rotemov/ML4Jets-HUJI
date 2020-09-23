from event_file_parser import EventFileParser
import numpy as np

"""
File the data was created from.
"""

R_VALUES = 0.4, 0.6, 0.8, 1.0

TRAINING_DATA_FILE_PATH = './Data/events_anomalydetection.h5'

for R in R_VALUES:
    parser = EventFileParser(TRAINING_DATA_FILE_PATH, '', R=R)
    parser.parse()
    # data is a numpy 2d array
    bg_events_array = np.array(parser.all_events['background'])
    file_name = './Data/Parsed_Data/bg_R{}'.format(R)
    np.save(file_name, bg_events_array)
    sig_events_array = np.array(parser.all_events['signal'])
    file_name = './Data/Parsed_Data/sig_R{}'.format(R)
    np.save(file_name, sig_events_array)

    print("Done")
