from __future__ import print_function, division
import math
import time
import os
import traceback


def _get_event_nsubjettines(outpute_file):
    output = open(outpute_file, "rb").read()
    lines = output.split("\n")

    def _single_space(s):
        while "  " in s:
            s = s.replace("  ", " ")
        return s

    jet1 = {}
    jet2 = {}
    for i in range(len(lines)):
        line = lines[i]
        if "Analyzing Jet 1" in line:
            beta_line = lines[i + 9]
            info_line = lines[i + 9 + 5]
            stripped_singles = _single_space(beta_line.strip()).split(" ")
            tau1, tau2 = stripped_singles[1:3]
            info = _single_space(info_line.strip()).split(" ")
            rap, phi, pt = info[1:4]
            e = info[5]
            jet1 = {'tau1': float(tau1), 'tau2': float(tau2), 'rap': float(rap), 'phi': float(phi), 'pt': float(pt), 'e': float(e)}
        if "Analyzing Jet 2" in line:
            beta_line = lines[i + 9]
            info_line = lines[i + 9 + 5]
            stripped_singles = _single_space(beta_line.strip()).split(" ")
            tau1, tau2 = stripped_singles[1:3]
            info = _single_space(info_line.strip()).split(" ")
            rap, phi, pt = info[1:4]
            e = info[5]
            jet2 = {'tau1': float(tau1), 'tau2': float(tau2), 'rap': float(rap), 'phi': float(phi), 'pt': float(pt), 'e': float(e)}
    print(jet1)
    print(jet2)
    return jet1, jet2


def _invmass(e, pt, eta):
    return e**2-(pt*math.cosh(eta))**2

#
# def get_subjettiness_fjcontrib(events_combined, m_total):
#     # # Get an event that fits what we want
#     # jets_in_range_indexes = [ i for i in range(len(m_total)) if 3000 <= m_total[i] <= 4000]
#
#     for event_col in events_combined:
#         get_fjcontrib_nsubjettiness(event_col)
#     print('Finished')


def get_fjcontrib_nsubjettiness(event_col):
    event_string = ''
    for j in range(700):
        if event_col[j * 3] > 0:
            pT = event_col[j * 3]
            eta = event_col[j * 3 + 1]
            phi = event_col[j * 3 + 2]

            px = pT * math.cos(phi)
            py = pT * math.sin(phi)
            pz = pT * math.sinh(eta)

            e = (px**2 + py**2 + pz**2)**0.5

            nv = [px, py, pz, e]
            event_string += '\t'.join(map(str, nv)) + '\n'

    epoch_time = int(time.time())
    file_name = 'python' + str(epoch_time) + '.dat'
    f = open("/Users/rotemmayo/Downloads/working_fastjet/fjcontrib-1.044/data/" + file_name, "w")
    f.write(event_string)
    f.close()
    output_filename = "output_" + file_name

    bashCommand = "cd /Users/rotemmayo/Downloads/working_fastjet/fjcontrib-1.044/Nsubjettiness ;" \
                  " ./my-short-example < ../data/" + file_name + " > ../data/" + output_filename
    try:
        os.system(bashCommand)
        jet1, jet2 = _get_event_nsubjettines(
            "/Users/rotemmayo/Downloads/working_fastjet/fjcontrib-1.044/data/" + output_filename)

        m1 = _invmass(jet1['e'], jet1['pt'], jet1['rap'])
        m2 = _invmass(jet2['e'], jet2['pt'], jet2['rap'])

        if m1 >= 0 and m2 >= 0:
            print(m1, m2)

    except:
        traceback.print_exc()
    finally:
        os.remove("/Users/rotemmayo/Downloads/working_fastjet/fjcontrib-1.044/data/" + output_filename)
        os.remove("/Users/rotemmayo/Downloads/working_fastjet/fjcontrib-1.044/data/" + file_name)
