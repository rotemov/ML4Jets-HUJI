from __future__ import print_function, division
import numpy as np
from pyjet import cluster, DTYPE_PTEPM
import pandas as pd
import math
import gc
from tqdm import tqdm

from event import Event
import jsonpickle
from serializable_psuedo_jet import SerializablePseudoJet


class EventFileParser:

    """
    Parses event files as given in the LHC olympics. Each line in an h5 file consists of maximum 700 partons with the
    coordinates (pt, eta, phi) and is zero padded. The 2101 cell contains a flag if it is signal or not in the
    development set.
    dev set should include ~100k signal ~1m bg.
    """
    def __init__(self, file_name, json_name, chunk_size=512, total_size=1100000, R=1.0, ptmin=20):
        """
        Creates an EventFileParser for the format of the files in the LHC olympics 2020
        :param file_name: the path to the file
        :param chunk_size: the number of lines it should handle at a time (this should help not to use too much memory)
        :param total_size: the number of lines in the file
        """
        self.file = EventFileParser.data_generator(file_name, chunk_size, total_size)
        self.chunksize = chunk_size
        self.total_size = total_size
        self.iterations = int(math.ceil(total_size / chunk_size))
        self.json_name = json_name
        self.R = R
        self.ptmin = ptmin
        self.all_events = {'background' : [], 'signal' : []}

    @staticmethod
    def data_generator(filename, chunksize, total_size):
        """
        Creates a generator of events from an h5 event file where each line is an event. Upon reaching the end of the
        file it will loop back to the start, meaning it doesn't indicate when file has ended.
        :param filename: The file path
        :param chunksize: The size of the chunks the generator outputs each time
        :param total_size: The total size of the file (number of lines)
        :return: A generator of the file that outputs chunks of chunk size
        """
        i = 0
        while True:
            yield pd.read_hdf(filename, start=i * chunksize, stop=(i + 1) * chunksize)
            i += 1
            if (i + 1) * chunksize > total_size:
                i = 0

    def parse(self):
        """
        Parses the file into a list of PsuedoJets of the events saved in the field alljets
        :return: None
        """
        for k in tqdm(range(self.iterations)):
            raw_events = self.file.next()
            n_events = np.shape(raw_events)[0]
            for index, event in raw_events.iterrows():
                issignal = (int(event[2100]) == 1)
                if issignal:
                    mytype = 'signal'
                else:
                    mytype = 'background'
                pseudojets_input = np.zeros(len([x for x in event[::3] if x > 0]), dtype=DTYPE_PTEPM)
                for j in range(700):
                    if (event[j * 3] > 0):
                        pseudojets_input[j]['pT'] = event[j * 3]
                        pseudojets_input[j]['eta'] = event[j * 3 + 1]
                        pseudojets_input[j]['phi'] = event[j * 3 + 2]
                        pass
                    pass
                sequence = cluster(pseudojets_input, R=self.R, p=-1)
                jets = sequence.inclusive_jets(ptmin=self.ptmin)
                sjets = [SerializablePseudoJet(j) for j in jets]
                event = Event(sjets, R=self.R)
                self.all_events[mytype] += [event.get_as_output()]
                gc.collect()
                pass
            # print("Chunk " + str(k) + " complete")


    #TODO: make this output a file and parse a file
    #TODO: make this ouput the parser itself might be even better
    def events_to_json(self):
        """
        Makes all events into a json format.
        :return: json format of all events
        """
        return jsonpickle.encode(self.all_events)

    @staticmethod
    def events_from_json(frozen):
        return jsonpickle.decode(frozen)

    # TODO: Add event to list function.
    # TODO: Add output_json_file function.
