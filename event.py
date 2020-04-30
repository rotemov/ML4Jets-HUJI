from __future__ import print_function, division
import numpy as np
from pyjet import cluster
import itertools

SUPPORTED_OBSERVALBES = ['m_tot', 'mjj', 'all_mjj', 'nj', 'm1', 'm2', 'm1_minus_m2', 'subjettiness', 'leading_pt',
                         'ht', 'mht']
DEFAULT_OBSERVABLES = ['m_tot', 'mjj', 'all_mjj', 'nj', 'm1', 'm2', 'm1_minus_m2', 'subjettiness']

# TODO: split subjettiness into different taus if it is needed and if it take up too much computation.


class Event:

    """
    Represents an event and all of it's observables.
    Contains methods to calculate all supported observables given a list of jets.
    """
    def __init__(self, jets, index=-1, box=-1, is_signal=False, observables_list=DEFAULT_OBSERVABLES):
        """
        Creates an event with all the supported observables calculated.
        :param jets: A list of the jets in the event.
        """
        self.jets = jets
        self.observables = {}
        self.__populate_observables(observables_list)
        self.jets_cart = []
        self.__poputlate_jets_cart()
        self.index = index
        self.box = box
        self.is_signal = is_signal

    def __poputlate_jets_cart(self):
        """
        Populates jets cart with the jet's cartesian coordinates.
        :param jets:
        :return:
        """
        for jet in self.jets:
            self.jets_cart += [jet.px, jet.py, jet.pz, jet.e]

    def __populate_observables(self, observables_list):
        """
        Populates the observables field.
        :param observables_list:
        :return:
        """
        for obs in observables_list:
            self.observables[obs] = Event.calc_observable(self.jets, obs)

    def get_observable(self, observable):
        """
        Given a name of an observable returns the observable's value for this event. See documentation of the different
        observables get functions in this class to understand output type.
        :param observable: The name of the observable. Must be in supported observables list.
        :return: The observable's value for this event.
        """
        if observable not in self.observables.keys():
            self.observables[observable] = Event.calc_observable(self.jets, observable)
        return self.observables[observable]

    @staticmethod
    def calc_observable(event_jets, obs):
        """
        Given a name of an observable returns the observable's value for this event. See documentation of the different
        observables get functions in this class to understand output type.
        :param event_jets: A list of jets from pyjet (PsuedoJet objects from pyjet cluster function)
        :param obs: The name of the observable. Must be in supported observables list.
        :return: The value of the observable.
        """
        if obs not in SUPPORTED_OBSERVALBES:
            raise Exception('Unsupported observable name please choose from list of supported observables:\n' +
                            str(SUPPORTED_OBSERVALBES))
        elif obs == 'm_tot':
            return Event.m_tot(event_jets)
        elif obs == 'mjj':
            return Event.mjj(event_jets)
        elif obs == 'all_mjj':
            return Event.get_mjj_all_pairs(event_jets)
        elif obs == 'nj':
            return Event.get_nj(event_jets)
        elif obs == 'm1':
            return Event.m1(event_jets)
        elif obs == 'm2':
            return Event.m2(event_jets)
        elif obs == 'm1_minus_m2':
            return Event.m1_minus_m2(event_jets)
        elif obs == 'subjettiness':
            return Event.get_event_nsubjettiness(event_jets)
        elif obs == 'leading_pt':
            return Event.get_lead_pt(event_jets)
        elif obs == 'ht':
            return Event.get_ht(event_jets)
        elif obs == 'mht':
            return Event.get_mht(event_jets)
        else:
            return None

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
    def m_tot(event_jets):
        """
        Finds the invariant mass of all the jets in an event.
        :param event_jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The invariant mass of the jets.
        """
        m_tot = Event.invariant_mass(event_jets)
        return m_tot

    @staticmethod
    def mjj(event_jets):
        """
        Finds the invariant mass of all the jets in an event.
        :param event_jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The invariant mass of the jets.
        """
        if len(event_jets) >= 2 :
            return Event.invariant_mass(event_jets[0:2])
        else:
            return Event.invariant_mass(event_jets)

    @staticmethod
    def m1(event_jets):
        """
        finds the invariant mass of the leading jet in an event.
        :param event_jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The invariant mass of the leading jet.
        """
        if len(event_jets) > 0:
            return event_jets[0].mass
        else:
            return 0

    @staticmethod
    def m2(event_jets):
        """
        finds the invariant mass of the second leading jet in an event.
        :param event_jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The invariant mass of the second leading jet.
        """
        if len(event_jets) > 1:
            return event_jets[1].mass
        else:
            return 0

    @staticmethod
    def m1_minus_m2(event_jets):
        """
        Finds m1-m2 observable.
        :param event_jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The invariant mass of the second leading jet.
        """
        return Event.m1(event_jets) - Event.m2(event_jets)

    @staticmethod
    def get_mjj_all_pairs(event_jets):
        mjj_all_pairs = []
        for pair in itertools.product(event_jets, repeat=2):
            (jo, jt) = pair
            mjj = Event.invariant_mass([jo, jt])
            if mjj >= 0:
                mjj_all_pairs += [mjj]
            else:
                raise Exception('Bad Mjj:' + str(mjj) + '\nJet 1:' + str(jo) + '\nJet 2:' + str(jt))
        return mjj_all_pairs

    @staticmethod
    def get_lead_pt(event_jets):
        """
        Gets the leading jet's pt from an event.
        :param event_jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The pt of the leading jet.
        """
        return event_jets[0].pt

    @staticmethod
    def get_nj(event_jets):
        """
        Get's the number of jets in an event
        :param event_jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :return: The number of jets in an event
        """
        return len(event_jets)

    @staticmethod
    def get_mht(event_jets, pt_cutoff=30, eta_cutoff=5):
        """
        Calculates the missing HT observable of an event.
        :param event_jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :param pt_cutoff: The cutoff pt
        :param eta_cutoff: The eta cutoff
        :return: The missing HT of an event
        """
        all_px = np.array([jet.px for jet in event_jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)])
        all_py = np.array([jet.py for jet in event_jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)])
        return sum(np.square(all_px) + np.square(all_py)) ** 0.5

    @staticmethod
    def get_ht(event_jets, pt_cutoff=30, eta_cutoff=2.5):
        """
        Calculates the HT observable of an event.
        :param event_jets: A list of all the jets from an even as given by pyjet cluster function (PsuedoJet objects).
        :param pt_cutoff: The cutoff pt
        :param eta_cutoff: The eta cutoff
        :return: The HT of an event
        """
        all_px = np.array([jet.px for jet in event_jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)])
        all_py = np.array([jet.py for jet in event_jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)])
        return sum(np.square(all_px) + np.square(all_py)) ** 0.5

    @staticmethod
    def get_nsubjettiness(jet, R=1.0, max_tau=4):
        """
        Calculate tau1, ..., tauN for the given jet (where N = max_tau)
        :param jet: The jet
        :param R: Radius to use when clustering
        :param max_tau: Highest tau to calculate
        :return: List containing all taus
        """
        particles = jet.constituents_array()
        num_particles = len(particles['eta'])
        sequence = cluster(particles, R=R, p=1)

        tau = np.zeros(max_tau)

        for i in range(1, min(max_tau + 1, num_particles)):
            sub_jets = sequence.exclusive_jets(i)

            delta_rs = np.zeros((num_particles, i))
            for j, sub_jet in enumerate(sub_jets):

                eta = particles['eta']
                phi = particles['phi']
                pi = np.pi

                if abs(sub_jet.phi) > pi / 2:
                    delta_rs[:, j] = np.sqrt(
                        (eta - sub_jet.eta) ** 2 + (phi % (2 * pi) - sub_jet.phi % (2 * pi)) ** 2)
                else:
                    delta_rs[:, j] = np.sqrt((eta - sub_jet.eta) ** 2 + (phi - sub_jet.phi) ** 2)

            delta_r = np.min(delta_rs, axis=1)
            tau[i - 1] = 1. / np.sum(particles['pT'] * R) * np.sum(particles['pT'] * delta_r)

        return list(tau)

    @staticmethod
    def get_event_nsubjettiness(event_jets, R=1.0, max_tau=4):
        """
        Given a list of an event's jets calculates their subjettiness.
        :param event_jets: All the jets in an event
        :param R: Anti-kt R parameter as needed for pyjet cluster function
        :param max_tau: The maximal subjettiness wanted
        :return: An array of arrays of jets subjettiness from the first jet to the last. i.e:
        [[jet1_tau1, jet1_tau2, ...], [jet2_tau1, jet2_tau2, ...], ...]
        """
        subjettiness = []
        for jet in event_jets:
            subjettiness += [Event.get_nsubjettiness(jet, R, max_tau)]
        return subjettiness

    # TODO: Add more observables
    # TODO: Sanity tests
