from pyjet import cluster
import numpy


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

    tau = numpy.zeros(max_tau)

    for i in range(1, min(max_tau + 1, num_particles)):
        sub_jets = sequence.exclusive_jets(i)

        delta_rs = numpy.zeros((num_particles, i))
        for j, sub_jet in enumerate(sub_jets):

            eta = particles['eta']
            phi = particles['phi']
            pi = numpy.pi

            if abs(sub_jet.phi) > pi / 2:
                delta_rs[:, j] = numpy.sqrt((eta - sub_jet.eta) ** 2 + (phi % (2 * pi) - sub_jet.phi % (2 * pi)) ** 2)
            else:
                delta_rs[:, j] = numpy.sqrt((eta - sub_jet.eta) ** 2 + (phi - sub_jet.phi) ** 2)

        delta_r = numpy.min(delta_rs, axis=1)
        tau[i - 1] = 1. / numpy.sum(particles['pT'] * R) * numpy.sum(particles['pT'] * delta_r)

    return list(tau)
