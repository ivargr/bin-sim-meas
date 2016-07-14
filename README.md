
# Quick overview

This is a collection of various small scripts used for investigating biases in binary similarity measures.

## Binary similarity measures (statistics)
76 binary similarity measures are implemented as methods in statistic_functions.py. These are typically passed into methods when running simulations.

## simulation_bias.py
For every binary sim meassure in a list, it simulates E and var for different track sizes. Plots the results.

Run example:

> $ python simuation_bias.py

## check_all_stats.py (should be renamed)
Contains the method check_negative_bias, which creates two sets of tracks (small and large tracks), and investigates whether one of the sets can win over the other sets that has large effect size by containing smaller tracks.

The resulting plot shows how much smaller the tracks in the "neutral" (no effect size) can be (needs to be) in order to rank higher than the larget tracks with given effect size.

Run example:

> $ python check_all_stats.py

(or run the method: check_negative_bias(stat_func = "name-of-statistic", n_sims=300, q_prob=1.0e-3, xmin=-3, xmax=0)

The file check_all_stats_api.php hacks a way to run this in paralell for multiple statistics.



