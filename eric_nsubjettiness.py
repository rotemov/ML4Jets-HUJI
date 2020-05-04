from pyjet import cluster
import numpy as np


def nsubjettiness(pseudojet, n, R=1, p=1):
    constituents = pseudojet.constituents_array()
    subjets = cluster(constituents, R=R, p=p).exclusive_jets(n)
    sumelem = [particle['pT']*min([np.sqrt((particle['phi'] - subjet.phi)**2+(particle['eta'] - subjet.eta)**2) for subjet in subjets]) for particle in constituents]
    tauN = sum(sumelem)/(sum(constituents['pT'])*R)
    return tauN
