"""
Using two sets of tracks. One set with small tracks, another with big tracsk. Same effect in both sets.
"""

from main import *
from statistic_functions import *

diffs = []


for func in statistic_functions:


    probs = [[1e-4, 3e-4], [1e-5, 3e-5]]

    s = Simulation(float(3e9), 5, float(1e-3))
    stat, winners, winners_stat = s.simulation_of_statistic(statistic_functions[func], probs, 1000)

    diff = (winners[1] - winners[0]) / float(winners[1] + winners[0])

    diffs.append([func, diff])

diffs = sorted(diffs, key=lambda t: abs(t[1]))

for diff in diffs:
    print "%.2f         on function %s" % (diff[1], diff[0])

#print diffs



