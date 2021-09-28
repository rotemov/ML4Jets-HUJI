from __future__ import print_function, division

import numpy as np
import itertools
import gc
from pyjet import cluster
from cached_property import cached_property


# TODO: split subjettiness into different taus if it is needed and if it take up too much computation.


class Event:
    """
    Represents an event and all of it's observables.
    Contains methods to calculate all supported observables given a list of jets.
    """

    def __init__(self, jets, index=-1, box=-1, is_signal=False, R=1.0):
        """
        Creates an event with all the supported observables calculated.
        :param jets: A list of the jets in the event.
        """
        self.jets = jets
        self.index = index
        self.box = box
        self.is_signal = is_signal
        self.R = R

    @cached_property
    def jets_cart(self):
        """
        Populates jets cart with the jet's cartesian coordinates.
        """
        return [[jet.px, jet.py, jet.pz, jet.e] for jet in self.jets]

    @staticmethod
    def invariant_mass(jets):
        """
        Finds the invariant mass of a list of jets.
        :param jets: A list of jets from pyjet (PsuedoJet objects from pyjet cluster function)
        :return: The invariant mass of the jets
        """
        e, px, py, pz = 0, 0, 0, 0
        for jet in jets:
            e += jet.e
            px += jet.px
            py += jet.py
            pz += jet.pz
        if (e ** 2 - px ** 2 - py ** 2 - pz ** 2) < 0:
            # We get a rounding error here, the number is just very close to zero
            return 0
        return (e ** 2 - px ** 2 - py ** 2 - pz ** 2) ** 0.5

    @cached_property
    def m_tot(self):
        """
        Finds the invariant mass of the event.
        :return: The invariant mass of the event.
        """
        return self.invariant_mass(self.jets)

    @cached_property
    def all_jet_mass(self):
        """
        Finds the invariant mass of all the jets in the event.
        :return: The invariant mass of the jets.
        """
        return [self.invariant_mass(jet) for jet in self.jets]

    @cached_property
    def all_jet_pt(self):
        """
        Finds the pt of all the jets in the event.
        :return: The invariant mass of the jets.
        """
        return [jet.pt for jet in self.jets]

    @cached_property
    def mjj(self):
        """
        Finds the invariant mass of all the jets in the event.
        :return: The invariant mass of the jets.
        """
        if len(self.jets) >= 2:
            return self.invariant_mass(self.jets[0:2])
        else:
            return self.invariant_mass(self.jets)

    @cached_property
    def m1(self):
        """
        finds the invariant mass of the leading jet in the event.
        :return: The invariant mass of the leading jet.
        """
        if len(self.jets) > 0:
            return self.invariant_mass([self.jets[0]])
        return 0

    @cached_property
    def m2(self):
        """
        finds the invariant mass of the second leading jet in the event.
        :return: The invariant mass of the second leading jet.
        """
        if len(self.jets) > 1:
            return self.invariant_mass([self.jets[1]])
        return 0

    @cached_property
    def m1_minus_m2(self):
        """
        Finds m1-m2 observable.
        :return: The invariant mass of the second leading jet.
        """
        return abs(self.m1 - self.m2)

    @cached_property
    def mjj_all_pairs(self):
        mjj_all_pairs = []
        for pair in itertools.product(self.jets, repeat=2):
            (jo, jt) = pair
            mjj_all_pairs += [self.invariant_mass([jo, jt])]
        return mjj_all_pairs

    @cached_property
    def lead_pt(self):
        """
        Gets the leading jet's pt from the event.
        :return: The pt of the leading jet.
        """
        return self.jets[0].pt

    @cached_property
    def nj(self):
        """
        Get's the number of jets in the event
        :return: The number of jets in an event
        """
        return len(self.jets)

    @cached_property
    def mht(self, pt_cutoff=30, eta_cutoff=5):
        """
        Calculates the missing HT observable of the event - the vector sum of the transverse momenta of all jets
        :param pt_cutoff: The cutoff pt
        :param eta_cutoff: The eta cutoff
        :return: The missing HT of an event
        """
        all_px = np.array([jet.px for jet in self.jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)])
        all_py = np.array([jet.py for jet in self.jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)])
        all_pz = np.array([jet.pz for jet in self.jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)])

        sum_px = sum(all_px)
        sum_py = sum(all_py)
        sum_pz = sum(all_pz)

        return (sum_px**2 + sum_py**2 + sum_pz**2) ** 0.5

    @cached_property
    def ht(self, pt_cutoff=30, eta_cutoff=2.5):
        """
        Calculates the HT observable of the event - scalar sum of the pt of all jets
        :param pt_cutoff: The cutoff pt
        :param eta_cutoff: The eta cutoff
        :return: The HT of an event
        """
        all_pts = [jet.pt for jet in self.jets if (jet.pt > pt_cutoff and jet.eta < eta_cutoff)]
        return sum(all_pts)

    @staticmethod
    def jet_nsubjettiness(jet, R=1.0, max_tau=4):
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

    @cached_property
    def nsubjettiness(self, max_tau=4):
        """
        Given a list of an event's jets calculates their subjettiness.
        :param R: Anti-kt R parameter as needed for pyjet cluster function
        :param max_tau: The maximal subjettiness wanted
        :return: An array of arrays of jets subjettiness from the first jet to the last. i.e:
        [[jet1_tau1, jet1_tau2, ...], [jet2_tau1, jet2_tau2, ...], ...]
        """
        return [self.jet_nsubjettiness(jet, self.R, max_tau) for jet in self.jets]

    @cached_property
    def all_tau21(self):
        """
        Gets the tau21 of all of the jets
        :return: The tau21 of all of the jets
        """
        return [nsubjetiness[1] / nsubjetiness[0] for nsubjetiness in self.nsubjettiness]

    """
    def __dict__(self):
        return {"index": self.index,
                "box": self.box,
                "mjj": self.mjj,
                "m1": self.m1,
                "m2": self.m2,
                "jets_cart": self.jets_cart}
    """

    def get_as_output(self):
        """
        :return: array containing: mjj, nj, m_tot, first 2 tau21
        """
        # return [self.mjj, self.nj, self.m_tot, self.m1, self.m2] + self.all_tau21[:2] # original obs
        return [self.mjj, self.m_tot, self.m1, self.m2, int(self.is_signal)]
        # return [self.mjj, self.nj, self.m_tot, self.m1, self.m2, self.m1_minus_m2, self.lead_pt, self.ht, self.mht] + \
        #        self.all_tau21[:2] + self.parton_data + [int(self.is_signal)]
        # return [self.nj, self.ht, self.mht, int(self.is_signal)]

    @cached_property
    def parton_data(self, n_jets=4, n_partons=10):
        """
        Returns a list of all the jets and their partons going by most energetic.
        @param n_jets: The number of jets to look at
        @param n_partons: The amount of partons to take from each jet
        """
        data_per_parton = 4
        coordinates = []
        labels = []
        for i, jet in enumerate(self.jets[:n_jets]):
            jet_coordinates = []
            for parton in jet.constituents[:n_partons]:
                parton_data = [parton[0], parton[1] - jet.eta, (parton[2] - jet.phi)%(2*np.pi), parton[3]]  # pt, delta eta, delta phi, mass
                jet_coordinates += parton_data
                labels.append(i)
            null_partons = n_partons - len(labels)
            jet_coordinates += [0] * null_partons * data_per_parton
            labels += [-1] * null_partons
            coordinates += jet_coordinates
            gc.collect()
        null_jets = int(n_jets - (len(coordinates) / (n_partons * data_per_parton)))
        coordinates += [0] * null_jets * n_partons * data_per_parton
        labels += [-1] * null_jets * n_partons
        return coordinates + labels

    # TODO: Add more observables
    # TODO: Sanity tests
