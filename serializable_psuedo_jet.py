class SerializablePseudoJet:
    """
    A serializable class that hold all the important fields of PseudoJet
    """
    def __init__(self, jet):
        self.px = jet.px
        self.py = jet.py
        self.pz = jet.pz
        self.e = jet.e
        self.pt = jet.pt
        self.eta = jet.eta
        self.phi = jet.phi
        self.constituents = jet.constituents_array()

    def constituents_array(self):
        """
        mimics the same method from PseudoJet
        :return:
        """
        return self.constituents
