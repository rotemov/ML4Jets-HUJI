from __future__ import print_function, division
from event import Event
from enum import Enum


class Observable(Enum):
    JETS_CART = 0
    M_TOT = 1
    ALL_JET_MASS = 2
    ALL_JET_PT = 3
    MJJ = 4
    M1 = 5
    M2 = 6
    M1_MINUS_M2 = 7
    MJJ_ALL_PAIRS = 8
    LEAD_PT = 9
    NJ = 10
    MHT = 11
    HT = 12
    ALL_JET_NSUBJETTINESS = 13
    ALL_TAU21 = 14
    EVENT_SUBJETTINESS = 15


class EventCSV(Event):
    """
    Represents an event and all of it's observables.
    Contains methods to calculate all supported observables given a list of jets.
    """

    def __init__(self, jets, obslist=[], index=-1, box=-1, is_signal=False,
                 mht_pt_cutoff=30, mht_eta_cutoff=5, ht_pt_cutoff=30, ht_eta_cutoff=2.5,
                 subjettiness_R=1.0, max_tau=4):
        """
        Creates an event with all the supported observables calculated.
        :param jets: A list of the jets in the event.
        """
        super(EventCSV, self).__init__(jets, index, box, is_signal)
        for obs in obslist:
            self._populate_observable(obs, mht_pt_cutoff, mht_eta_cutoff, ht_pt_cutoff, ht_eta_cutoff,
                 subjettiness_R, max_tau)

    def _populate_observable(self, obs, mht_pt_cutoff, mht_eta_cutoff, ht_pt_cutoff, ht_eta_cutoff,
                 subjettiness_R, max_tau):
        """
        Given an observable populates it's property.
        :param obs:
        :param mht_pt_cutoff:
        :param mht_eta_cutoff:
        :param ht_pt_cutoff:
        :param ht_eta_cutoff:
        :param subjettiness_R:
        :param max_tau:
        :return:
        """
        val = obs.value
        if val == Observable.JETS_CART.value:
            self.jets_cart = super(EventCSV, self).jets_cart()
        elif val == Observable.M_TOT.value:
            self.m_tot = super(EventCSV, self).m_tot()
        elif val == Observable.ALL_JET_MASS.value:
            self.all_jet_mass = super(EventCSV, self).all_jet_mass()
        elif val == Observable.ALL_JET_PT.value:
            self.all_jet_pt = super(EventCSV, self).all_jet_pt()
        elif val == Observable.MJJ.value:
            self.mjj = super(EventCSV, self).mjj()
        elif val == Observable.M1.value:
            self.m1 = super(EventCSV, self).m1()
        elif val == Observable.M2.value:
            self.m2 = super(EventCSV, self).m2()
        elif val == Observable.M1_MINUS_M2.value:
            self.m1_minus_m2 = super(EventCSV, self).m1_minus_m2()
        elif val == Observable.MJJ_ALL_PAIRS.value:
            self.mjj_all_pairs = super(EventCSV, self).mjj_all_pairs()
        elif val == Observable.LEAD_PT.value:
            self.lead_pt = super(EventCSV, self).lead_pt()
        elif val == Observable.NJ.value:
            self.nj = super(EventCSV, self).nj()
        elif val == Observable.MHT.value:
            self.mht = super(EventCSV, self).mht(mht_pt_cutoff, mht_eta_cutoff)
        elif val == Observable.HT.value:
            self.ht = super(EventCSV, self).ht(ht_pt_cutoff, ht_eta_cutoff)
        elif val == Observable.ALL_JET_NSUBJETTINESS.value:
            self.all_jet_nsubjettiness = super(EventCSV, self).all_jet_nsubjettiness(subjettiness_R, max_tau)
        elif val == Observable.ALL_TAU21.value:
            self.all_tau21 = super(EventCSV, self).all_tau21()
        elif val == Observable.EVENT_SUBJETTINESS.value:
            self.event_subjettiness = super(EventCSV, self).event_subjettiness(subjettiness_R, max_tau)

           
