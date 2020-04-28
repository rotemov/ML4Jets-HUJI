from __future__ import print_function, division
import numpy as np
from pyjet import cluster,DTYPE_PTEPM
import pandas as pd
import math
import time
import os
import traceback
import matplotlib.pyplot as plt



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
    return jet1, jet2


def _invmass(e, pt, eta):
    return e**2-(pt*math.cosh(eta))**2


# Constants
R = 0.4
p = -1
FILE_PATH = "/Users/rotemmayo/Documents/PyCharm/Data/events_LHCO2020_BlackBox1.h5"
NUMBER_OF_EVENTS = 1000
TRY_SUBJETTINES = False


# Read File
print('Read File')
fnew = pd.read_hdf(FILE_PATH, stop=NUMBER_OF_EVENTS)
events_combined = fnew.T


# Cluster jets
print('Cluster Jets')
alljets = []
for i in range(np.shape(events_combined)[1]):
    if i % (NUMBER_OF_EVENTS / 10) == 0:
        print(i)

    pseudojets_input = np.zeros(len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
    is_signal = True
    # events_combined[i][2100]
    if is_signal:
        for j in range(700):
            if events_combined[i][j * 3] > 0:
                pseudojets_input[j]['pT'] = events_combined[i][j * 3]
                pseudojets_input[j]['eta'] = events_combined[i][j * 3 + 1]
                pseudojets_input[j]['phi'] = events_combined[i][j * 3 + 2]

        sequence = cluster(pseudojets_input, R=R, p=p)
        jets = sequence.inclusive_jets(ptmin=20)
        alljets += [jets]

print('DONE ', len(alljets))


x = []
y = []
for k in range(len(alljets)):
    jet1 = alljets[k][0]
    jet2 = alljets[k][1]
    m_jet1 = (jet1.e**2-jet1.px**2-jet1.py**2-jet1.pz**2)**0.5
    m_jet2 = (jet2.e**2-jet2.px**2-jet2.py**2-jet2.pz**2)**0.5

    y += [abs(m_jet1 - m_jet2)]
    x += [max(m_jet1, m_jet2)]


fig = plt.figure()
plt.scatter(x, y)
plt.xlabel('m1')
plt.ylabel('m1-m2')
plt.show()


# Calculate m total for all jets
print('m total')
m_total = []
for i in range(len(alljets)):
    jets = alljets[i]
    all_px = sum([j.px ** 2 for j in jets])
    all_py = sum([j.py ** 2 for j in jets])
    all_pz = sum([j.pz ** 2 for j in jets])
    all_e = sum([j.e for j in jets])

    if all_e ** 2 - all_px - all_py - all_pz > 0:
        m_total += [(all_e ** 2 - all_px - all_py - all_pz) ** 0.5]


if TRY_SUBJETTINES:
    # Get an event that fits what we want
    jets_in_range_indexes = [ i for i in range(len(m_total)) if 3000 <= m_total[i] <= 4000]

    for k in range(len(jets_in_range_indexes)):
        event_col = events_combined[jets_in_range_indexes[k]]
        event_string = ''
        for j in range(700):
            if event_col[j*3]>0:
                E = event_col[j*3+3]
                pT = event_col[j*3]
                eta = event_col[j*3+1]
                phi = event_col[j*3+2]

                # [px, py, pz, E]
                nv = [pT*math.cos(phi), pT*math.sin(phi), pT*math.sinh(eta), E]
                event_string += '\t'.join(map(str, nv)) + '\n'

        epoch_time = int(time.time())
        file_name = 'python' + str(epoch_time) + '.dat'
        f = open("/Users/rotemmayo/Downloads/working_fastjet/fjcontrib-1.044/data/" + file_name, "w")
        f.write(event_string)
        f.close()
        output_filename = "output_" + file_name

        bashCommand = "cd /Users/rotemmayo/Downloads/working_fastjet/fjcontrib-1.044/Nsubjettiness ;" \
                      " ./short-example2 < ../data/" + file_name + " > ../data/" + output_filename
        try:
            os.system(bashCommand)
            jet1, jet2 = _get_event_nsubjettines("/Users/rotemmayo/Downloads/working_fastjet/fjcontrib-1.044/data/" + output_filename)

            m1 = _invmass(jet1['e'], jet1['pt'], jet1['rap'])
            m2 = _invmass(jet2['e'], jet2['pt'], jet2['rap'])

            if m1 >= 0 and m2 >= 0:
                print(k, m1, m2)

            # if len(jet1) != 0:
            #     print(k, _invmass(jet1['e'], jet1['pt'], jet1['rap']))
            # else:
            #     print(k)
            # if len(jet2) != 0:
            #     print(k, _invmass(jet2['e'], jet2['pt'], jet2['rap']))
            # else:
            #     print(k)
        except:
            traceback.print_exc()
        finally:
            os.remove("/Users/rotemmayo/Downloads/working_fastjet/fjcontrib-1.044/data/" + output_filename)
            os.remove("/Users/rotemmayo/Downloads/working_fastjet/fjcontrib-1.044/data/" + file_name)

    print('Finished', k)
