from main import *
from statistic_functions import *
import numpy as np
#import matplotlib.pyplot as plt

from scipy.stats import hypergeom
from scipy.stats import binom

def get_d(winners):
    if winners[1]<winners[0]:
        return 1
    if winners[1]>winners[0]:
        return -1
    else:
        return 0 


class SimulationParams(object):
    def __init__(self, probs, n_sims, set_size):
        self.probs = probs
        self.n_sims = n_sims
        self.set_size = set_size

class RegrassionParams(object):
    def __init__(self, x_min,x_max, y_min,y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


set_size = 30
def find_optimal_size_grid(p, p1,stat_func,k=2,xmin=-4,xmax=4, n_sims = 100, q_prob = 1.0e-3):
    size_factor = 0.5
    sim_annel = 0.5
    cnt = 0
    win_ratio = np.zeros(200)
    prev_d = None
    ys = np.power(10, np.linspace(xmin,xmax, 320))
    w_factor = np.zeros(len(ys))
    winners_stats = np.zeros(len(ys))
    for size_factor in ys:
        S = Simulation(float(3e9), set_size, q_prob)
        _, winners, winners_stat = S.simulation_of_statistic(stat_func, [[p,  k*p1], [size_factor*p, k*size_factor*p]], n_sims)
        w_factor[cnt] = winners[0]-winners[1]
        winners_stats[cnt] = abs(winners_stat[0] - winners_stats[1])
        cnt = cnt+1

    i = np.argmin(abs(w_factor))
    #plt.plot(ys, w_factor)
    #plt.show()
    if (np.min(w_factor)<=0 and np.max(w_factor)>=0):
        return ys[i]
    return np.NaN

def find_optimal_size_converge(p, p1,stat_func,k=2,y_min=-4,y_max=4, seed = 0, n_sims=100, q_prob = 1.0e-3):
    #y_max=np.log10(min(1/(k*p),np.log10(y_max)))

    #y_max=min(-np.log10(p) - np.log10(k + 1/q_prob - 1), y_max) # - np.log10(k*p), y_max)
    y_max=min(np.log10(q_prob) - np.log10(k*p), y_max)
    #print "yMax %.3f" % y_max

    S = Simulation(float(3e9), 10, q_prob)
    sf_min = np.power(10,y_min)
    sf_max = np.power(10,y_max)
    
    win_func = lambda x: (x[0]-x[1])/(x[0]+x[1])

    _, winners_min, asdf = S.simulation_of_statistic(stat_func, [[p,  k*p1], [sf_min*p, sf_min*k*p]],n_sims*10)
    _, winners_max, asdf = S.simulation_of_statistic(stat_func, [[p,  k*p1], [sf_max*p, sf_max*k*p]],n_sims*10)
    
    w_score_min = win_func(winners_min)
    w_score_max = win_func(winners_max)
    if w_score_max>=0 and w_score_min<=0:
        r = 1
    elif w_score_max<=0 and w_score_min>=0:
        r= -1
    else:
        return find_optimal_size_grid(p,p1,stat_func,k,y_min,y_max,n_sims, q_prob),0

        #return np.NaN , 0#TODO

    sf  = np.power(10, seed)
    t = 1
    cooling_rate=0.7
    ydiff = y_max-y_min
    min_val = 1
    for i in xrange(10):
        _, winners, _ = S.simulation_of_statistic(stat_func, [[p,  k*p1], [sf*p, sf*k*p]],n_sims)
        #print "Seed: %.4f, Winners: %s" % (seed, str(winners))
        wscore = win_func(winners)
        min_val = min(abs(wscore), min_val)
        if r == 0:
            seedn = min(max(seed-0.5*ydiff*t*wscore*(-1),y_min) ,y_max)
            seedp = min(max(seed-0.5*ydiff*t*wscore*(1),y_min) ,y_max)
            sfn = np.power(10, seedn)
            sfp = np.power(10, seedp)
            _, winnersn, _ = S.simulation_of_statistic(stat_func, [[p,  k*p1], [sfn*p, sfn*k*p]],n_sims)
            _, winnersp, _ = S.simulation_of_statistic(stat_func, [[p,  k*p1], [sfp*p, sfp*k*p]],n_sims)
            if abs(win_func(winnersn))>abs(win_func(winnersp)):
                sf = sfp
                seed = seedp
            else:
                sf = sfn
                seed = seedn
        else:
            seed = min(max(seed-0.5*ydiff*t*wscore*(-1),y_min) ,y_max)
            sf = np.power(10,seed)

        t = t*cooling_rate
    if min_val>0.1:
        return np.NaN, 0
    return sf,seed


def plot_size_vs_effect(sizes, effects,stat_name):
    import numpy as np

    # Hack
    print sizes
    for i in range(0, len(sizes)):
        #if type(sizes[i]).__module__ == "numpy":
        if isinstance(sizes[i], np.ndarray):
            sizes[i] = sizes[i][0]


    print sizes
    plt.plot(effects, np.log10(sizes))
    plt.xlim((0.5, 3))
    plt.ylim((-0.5, 4))
    plt.xlabel("Effect size")
    plt.ylabel("log(Track size ratio)")
    plt.title(stat_name)
    plt.savefig(FOLDER + "/" + stat_name+'.png', bbox_inches='tight')
    plt.close()



def plot_size_vs_effect_on_figure(figure, legend, sizes, effects,stat_name, xmin, xmax):
    import numpy as np

    # Hack: Some sizes are numpy arrays
    #print sizes
    for i in range(0, len(sizes)):
        #if type(sizes[i]).__module__ == "numpy":
        if isinstance(sizes[i], np.ndarray):
            sizes[i] = sizes[i][0]


    #print sizes
    plt.plot(effects, np.log10(sizes))
    plt.xlim((1, 3))
    plt.ylim((xmin, xmax))
    plt.xlabel("Effect size")
    plt.ylabel("Track size ratio")
    #plt.title(stat_name)
    #plt.legend(legend)

def get_bias_functionC(name, func,xmin=-4, xmax=4, n_sims=100, q_prob = 1.0e-3):
    size_factors = []
    effect_sizes = np.linspace(1, 3, 20)
    seed = 0
    for effect_size in effect_sizes:
        p = 1e-4
        p1 = 1e-4 * effect_size

        # Run on grid
        size = find_optimal_size_grid(p,p1,func,xmin=xmin,xmax=xmax, n_sims=n_sims, q_prob=q_prob)

        # Use converging search:
        #size,seed = find_optimal_size_converge(p,p1,func,y_min=xmin,y_max=xmax,seed = seed, n_sims=n_sims, q_prob=q_prob)


        print "Progress: %.1f " % (100 * (effect_size -1) / (3.0 - 1))
        size_factors.append(size)
    return effect_sizes,size_factors

"""
def get_bias_function(name, func,xmin=-4, xmax=4, n_sims = 100):
    size_factors = []
    effect_sizes = np.linspace(0.5, 3, 50)
    for effect_size in effect_sizes:
        p = 1e-4
        p1 = 1e-4 * effect_size
        size = find_optimal_size_grid(p,p1,func,xmin=xmin,xmax=xmax, n_sims=n_sims)
        print "Effect size: %d" % (effect_size)
        size_factors.append(size)
    return effect_sizes,size_factors
"""

def write_data(f, name, effect_sizes, size_factors, ymin, ymax, nsims, q_prob):
    f.write("%sEffect: %s\n"%(name,effect_sizes))
    f.write("%sSizeRatio: %s\n"%(name,size_factors))
    np.savetxt(FOLDER + "/raw_data_stat=%s_ymin=%d_ymax=%d_nsims=%d_qprob=%.1E_setsize=%d" % (name, ymin, ymax, nsims, q_prob, set_size), [size_factors, effect_sizes])

def read_data(f,):
    text = f.read()
    parts = text.split(":")
    names = []
    effect_sizes = []
    size_factors = []
    name = ""
    effect=""
    ratio =""
    for i,part in enumerate(parts):
        if (i==0):
            name = part.replace("Effect","")
            continue

        parts2 = part.split("]")
        if (i%2==1):
            effect = [float(x) for x in  parts2[0].replace("[","").strip().split()]
            name = parts2[1].strip().replace("SizeRatio","")
        else:
            ratio = [float(x.strip()) for x in  parts2[0].replace("[","").split(",")]
        if (i%2==0):
            names.append(name)
            effect_sizes.append(effect)
            size_factors.append(ratio)
   
    return names, effect_sizes,size_factors


def check_negative_bias(stat_func = "", n_sims = 100, q_prob = 1.0e-3, xmin=-4, xmax=0):
    f = open('sizeandeffectNeg.txt', 'w')

    if stat_func == "":
        closer = ["cosine"]#, "driver-kroeber", "eyraud", "forbes1", "forbes2", "gilbert-wells",  "johnson", "kulczynski2","mcconnaughey", \
        #"mountford", "pearson and heron-II","sample", "simpson", "sokal-sneath4", "tarantula", "tarwid", "YULEw", "t6"]
    else:
        closer = [stat_func]
    #closer = ["driver-kroeber"]


    #closer = ["driver-kroeber", "kulczynski2", "sokal-sneath4", "forbes2", "forbes1", "YULEQ"]
    #closer = ["aoverbnorm", "driver-kroeber"]# "kulczynski2", "sokal-sneath4", "forbes2", "forbes1", "YULEQ"]
    #closer = ["cosine", "forbes1", "gilbert-wells", "pearson and heron-II", "YULEQ", "YULEw", "tarwid", "sokal-sneath4"]
    #closer = ["cosine", "sokal-sneath4", "YULEw", "YULEQ", "tarwid"]
    #closer = ["tarwid", "sokal-sneath4", "driver-kroeber", "mcconnaughey", "kulczynski2"]
    #closer = ["tarwid", "sokal-sneath4", "t5"]
    #closer = ["t5", "sokal-sneath4"]

    figure = plt.figure()

    for name in closer:
        stat = statistic_functions[name]
        print "---Trying, " + name
        effect_sizes,size_factors = get_bias_functionC(name, stat,xmin = xmin,xmax=xmax, n_sims = n_sims, q_prob=q_prob)
        #plot_size_vs_effect(size_factors, effect_sizes, name)
        plot_size_vs_effect_on_figure(figure, name, size_factors, effect_sizes, name, xmin, xmax)
        #plot_size_vs_effect(size_factors, effect_sizes, name+"neg")
        write_data(f, name,  effect_sizes,size_factors, xmin, xmax, n_sims, q_prob)

    plt.legend(closer)
    plt.savefig(FOLDER + "/all_%d_sims_qprob_%.2E_set_size_%d.png" % (n_sims, q_prob, set_size), bbox_inches='tight')
    plt.show()
    #plt.close()

    f.close()

"""
def check_pos_bias():
    f = open('sizeandeffectPos.txt', 'w')
    good = ["YULEQ"] #,"anderberg","goodman and kruskal"]
    for name in good:
        stat = statistic_functions[name]
        print "---Trying, " + name
        try:
            effect_sizes,size_factors = get_bias_function(name, stat,xmin = 0,xmax=4)
        except:
            print name + " failed"
            continue
        plot_size_vs_effect(size_factors, effect_sizes, name+"pos")
        write_data(f, name,  effect_sizes,size_factors)
    f.close()
"""

def visualize_written_data():
    f = open('sizeandeffect.txt', 'r')
    names, effects,ratios = read_data(f)
    f.close()
    for n,e,r in zip(names,effects,ratios):
        plot_size_vs_effect(s, e, n)



if __name__ =="__main__":

    set_size = 10 # Global
    FOLDER = "figures7"
    check_negative_bias(stat_func = "forbes1", n_sims=30, q_prob=1.0e-3, xmin=-3, xmax=0)
    #check_negative_bias(stat_func = "jaccard", n_sims=50, q_prob=1.0e-3, xmin=-4, xmax=0)
    #check_negative_bias(stat_func = "jaccard", n_sims=300, q_prob=1.0e-3, xmin=0, xmax=5)
#    check_pos_bias()
    #f = open('sizeandeffectNeg.txt', 'r')
    #names, effects,ratios = read_data(f)
    #f.close()
    #for n,e,r in zip(names,effects,ratios):
    #    plot_size_vs_effect(r, e, n)

    good = ["YULEQ","anderberg","goodman and kruskal", "aoverbnorm"]


#    
#
#
#    f = open('sizeandeffect.txt', 'w')
#    for name,stat in statistic_functions.iteritems():
#        print "---Trying, " + name
#        try:
#            effect_sizes,size_factors = get_bias_function(name, stat)
#        except:
#            print name + " failed"
#            continue
#        plot_size_vs_effect(size_factors, effect_sizes, name)
#        write_data(f, name,  effect_sizes,size_factors)
#    f.close()
