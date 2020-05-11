import nsubjettiness

from get_jets import get_jets_from_black_box

from nsubjettines_fjcontrib import get_fjcontrib_nsubjettiness

R = 1.0
NUMBER_OF_EVENTS = 1000


def compare_george_to_contrib():
    events, events_combined = get_jets_from_black_box(1, NUMBER_OF_EVENTS, R=R)

    print('### FJ CONTRIB ###')
    get_fjcontrib_nsubjettiness(events_combined[0])

    print('### GEORGE ###')
    for jet in events[0].jets:
        print(str(jet) + ' energy: ' + str(jet.e))
        print('tau1: %f, tau2: %f, tau3: %f, tau4: %f\n' % tuple(nsubjettiness.get_nsubjettiness(jet, R=R)))


compare_george_to_contrib()
