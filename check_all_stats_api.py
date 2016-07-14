#! /usr/bin/env python
import sys
from check_all_stats import *
import numpy as np

func_name = sys.argv[1]

n_sims = 50

if(len(sys.argv) > 2):
    n_sims = int(sys.argv[2])
print "Running function " + func_name
check_negative_bias(func_name, n_sims)