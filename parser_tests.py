from event_file_parser import EventFileParser


def tiny_data_test():
    """
    Sanity test for the parser.
    :return:
    """
    p = EventFileParser("/home/rotemov/PycharmProjects/ML4Jets-HUJI/Data/events_anomalydetection_tiny.h5", 100, 1000)
    p.parse()
    print(len(p.alljets['background']), len(p.alljets['signal']))
    j1 = p.alljets['background'][0]
    print(j1)
    print(EventFileParser.m_tot(j1))
    print(EventFileParser.m1_minus_m2(j1))
    j2 = p.alljets['signal'][0]
    print(j2)
    print(EventFileParser.m_tot(j2))
    print(EventFileParser.m1_minus_m2(j2))
    edge_cases()


def edge_cases():
    """
    Some tests for edge cases
    Cases covered: empty jet lists
    :return:
    """
    assert (EventFileParser.m1([]) == 0)
    assert (EventFileParser.m2([]) == 0)
    assert (EventFileParser.invariant_mass([]) == 0)
    assert (EventFileParser.m_tot([]) == 0)


tiny_data_test()