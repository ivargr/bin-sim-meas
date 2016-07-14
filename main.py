import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import hypergeom
from scipy.stats import binom
from test_statistics import *
from statistic_functions import *

class Simulation:

    def __init__(self, n_basepairs=3e9, set_size=10, q_prob=1e-3):
        self.n_basepairs = n_basepairs
        self.set_size = set_size
        self.q_prob = q_prob
        self.q_size = n_basepairs*self.q_prob

    # Computes a, b, c, d and runs a statistic function
    def statistic_function_wrapper(self, stat_func, R):
        a = R[:,0].astype(float)
        b = R[:,1].astype(float)
        c = self.q_size - R[:,0].astype(float)
        d = self.n_basepairs-a-b-c
        n = self.n_basepairs

        d = d.astype(float)



        return stat_func(a, b, c, d, n)

    # Simualtion of a statistic
    # probs is a list of list. The inner list contains the overall prob of the track and the prob of being inside Q
    def simulation_of_statistic(self, stat_func, probs, n_sim = 100):

        winners = np.zeros((len(probs), 1))
        winners_stat = np.zeros((len(probs), 1))
        stats = np.zeros((n_sim, self.set_size * len(probs)))
        #print "Statistics %s" % (stat_func.__name__)

        for i in range(0,n_sim):

            # Compute intersection with Q and Q^C for all Ri tracks
            R = np.zeros((self.set_size * len(probs), 4))
            j = 0
            for prob in probs:

                # Limit too high probabilities
                #prob = np.maximum([1.0, 1.0], prob)
                prob[1] = min(prob[1], 1.0)
                prob[0] = min(prob[0], 1.0)

                for set in range(0, self.set_size):
                    R[j, 0] = np.random.binomial(self.q_size, prob[1]) # The number of matches inside track Q
                    R[j, 1] = np.random.binomial(self.n_basepairs - self.q_size, prob[0]) # The number of matches outside track Q
                    R[j, 2] = R[j, 0] + R[j, 1]
                    R[j, 3] = prob[1]
                    j += 1



            # Find highest ranking track
            if stat_func == T5 or stat_func == T6 or stat_func == pval:
                stat = stat_func(R, self.n_basepairs, self.q_size) # Old way of calling stat directly (without wrapper)
            else:
                stat = self.statistic_function_wrapper(stat_func, R) # Old way of calling stat directly (without wrapper)

            # Hack: Some stats do not handle zero track size and will give nan or inf
            stat[np.isnan(stat)] = -1000
            stat[np.isinf(stat)] = -1000

            stats[i] = stat

            max_stat = np.argmax(stat) / self.set_size
            winners[max_stat] += 1
            winners_stat[max_stat] += np.argmax(stat)

        return stats, winners, winners_stat

    def plot_stats(self, stats):
        print "Variance and mean of statistic:"
        for i in range(0, len(probs)):
            statistics = np.matrix.flatten(stats[:, i * self.set_size : (i+1) * self.set_size])
            plt.scatter(np.ones((len(statistics), 1)) * np.log(probs[i][1]), statistics, 1, alpha=0.5)
            print " Tracks with prob (%.7f, %.7f). Mean: %.10f, variance: %.10f" % (probs[i][0], probs[i][1], np.mean(statistics), np.var(statistics))
            fig = plt.plot()
            #plt.hist(statistics, 50)
            #plt.show()
            #print statistics

        plt.show()


    def print_winners(self, winners):
        print "Best ranking:"
        for i in range(0, len(probs)):
            print " Number of times tracks with prob (%.7f, %.7f) won: %d" % (probs[i][0], probs[i][1], winners[i])



if __name__ == "__main__":
    # Simple test
    probs = [[1e-4, 1e-4], [1e-5, 1.1e-5], [1e-6, 1.1e-6]]
    s = Simulation(float(6e9), 20, float(1e-3))
    stats, winners, winners_stat = s.simulation_of_statistic(T6, probs, 500)
    s.plot_stats(stats)
    s.print_winners(winners)




