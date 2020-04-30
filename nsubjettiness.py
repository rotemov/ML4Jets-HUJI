from pyjet import cluster
import numpy


def nsubjettiness(jet, R=1.0, max_tau=4):
    particles = jet.constituents_array()
    num_particles = len(particles['eta'])
    sequence = cluster(particles, R=R, p=1)

    tau = numpy.zeros(max_tau)

    for i in range(1, max(max_tau + 1, num_particles)):
        sub_jets = sequence.exclusive_jets(i)

        delta_rs = numpy.zeros((num_particles, i))
        for j, subj in enumerate(sub_jets):

            if abs(subj.phi) > numpy.pi / 2:
                delta_rs[:, j] = numpy.sqrt((particles['eta'] - subj.eta) ** 2 +
                                            (particles['phi'] % (2 * numpy.pi) - subj.phi % (2 * numpy.pi)) ** 2)
            else:
                delta_rs[:, j] = numpy.sqrt((particles['eta'] - subj.eta) ** 2 +
                                            (particles['phi'] - subj.phi) ** 2)

        delta_r = numpy.min(delta_rs, axis=1)

        tau[i - 1] = 1. / numpy.sum(particles['pT'] * R) * numpy.sum(particles['pT'] * delta_r)

    return list(tau)
