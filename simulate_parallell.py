#!/usr/local/bin/python
from subprocess import Popen
from check_all_stats import *

closer = ["cosine", "driver-kroeber", "eyraud", "forbes1", "forbes2", "gilbert-wells",  "johnson", "kulczynski2","mcconnaughey", \
              "mountford", "pearson and heron-II","sample", "simpson", "sokal-sneath4", "tarantula", "tarwid", "YULEw", "t6"]


closer = ["cosine", "driver-kroeber", "eyraud", "forbes1", "forbes2", "gilbert-wells", "kulczynski2","mcconnaughey", \
              "mountford", "pearson and heron-II","sample", "simpson", "sokal-sneath4", "tarantula", "tarwid", "YULEw", "t6"]


closer = ["driver-kroeber", "kulczynski2", "sokal-sneath4", "forbes2", "forbes1"]

for func in closer:
    print "Running command with " + func

    path = "/home/ivar/dev/simulering/"
    Popen("/home/ivar/simulering2/gsuit-stat-simulations/check_all_stats_api.py " + func + " 10", shell=True)