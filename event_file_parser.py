from __future__ import print_function, division
import numpy as np
from pyjet import cluster,DTYPE_PTEPM
import pandas as pd
import math


class EventFileParser:

    def __init__(self, file_name, chunk_size=512, total_size=1100000):
        """
        Creates an EventFileParser for the format of the files in the LHC olympics 2020
        :param file_name: the path to the file
        :param chunk_size: the number of lines it should handle at a time (this should help not to use too much memory)
        :param total_size: the number of lines in the file
        """
        self.file = EventFileParser.data_generator(file_name, chunk_size, total_size)
        self.chunksize = chunk_size
        self.total_size = total_size
        self.iterations = math.ceil(total_size / chunk_size)
        self.alljets = {'background' : [], 'signal' : []}

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
        for k in range(self.iterations):
            raw_events = (self.file.__next__())
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
                sequence = cluster(pseudojets_input, R=1.0, p=-1)
                jets = sequence.inclusive_jets(ptmin=20)
                self.alljets[mytype] += [jets]
                pass
            print("Chunk " + str(k) + " complete")


    @staticmethod
    def invariant_mass(jets):
        """
        Finds the invariant mass of a list of jets.
        :param jets: A list of jets from pyjet (PsuedoJet objects from pyjet cluster function)
        :return: The invariant mass of the jets
        """
        E, px, py, pz = 0, 0, 0, 0
        for jet in jets:
            E += jet.e
            px += jet.px
            py += jet.py
            pz += jet.pz
        mass = (E**2 - px**2 - py**2 - pz**2)**0.5
        return mass

    @staticmethod
    def m_tot(jets):
        """
        Finds the invariant mass of all the jets in an event.
        :param jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The invariant mass of the jets.
        """
        m_tot = EventFileParser.invariant_mass(jets)
        return m_tot

    @staticmethod
    def mjj(jets):
        """
        Finds the invariant mass of all the jets in an event.
        :param jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The invariant mass of the jets.
        """
        if len(jets) >= 2 :
            return EventFileParser.invariant_mass(jets[0:2])
        else:
            return EventFileParser.invariant_mass(jets)

    @staticmethod
    def m1(jets):
        """
        finds the invariant mass of the leading jet in an event.
        :param jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The invariant mass of the leading jet.
        """
        if len(jets) > 0:
            return jets[0].mass
        else:
            return 0

    @staticmethod
    def m2(jets):
        """
        finds the invariant mass of the second leading jet in an event.
        :param jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The invariant mass of the second leading jet.
        """
        if len(jets) > 1:
            return jets[1].mass
        else:
            return 0

    @staticmethod
    def m1_minus_m2(jets):
        """
        Finds m1-m2 observable.
        :param jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The invariant mass of the second leading jet.
        """
        return EventFileParser.m1(jets)-EventFileParser.m2(jets)

    # TODO: Add n-subjettiness of different levels. Add more relevant observables.
    # TODO: Add output_h5_file function.
