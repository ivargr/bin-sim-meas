"""
For every statistic, simulates E and var for different track sizes. Plots the results.
"""
from main import Simulation
from statistic_functions import *
import matplotlib.pyplot as plt

FOLDER = ""

def get_data(stat_func, p,n_sims=1000,k=1):
    s = Simulation(set_size=1, q_prob = 1e-3)
    stats, _,_ = s.simulation_of_statistic(stat_func, [[p,k*p]],n_sims)
    var = np.var(stats)
    E = np.mean(stats)
    return E, var


def simulate_stat_by_k(stat_func, ks, n_sims = 1000, p=1e-4):
    Es = np.zeros(len(ks))
    vars_ = np.zeros(len(ks))
    for i,k in enumerate(ks):
        E,var = get_data(stat_func, p, n_sims,k)
        Es[i] = E
        vars_[i] = var

    return Es, vars_

def simulate_stat(stat_func, xs, n_sims = 1000, k=1):
    Es = np.zeros(len(xs))
    vars_ = np.zeros(len(xs))
    for i,x in enumerate(xs):
        E,var = get_data(stat_func, x, n_sims,k)
        Es[i] = E
        vars_[i] = var

    return Es, vars_

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

def plot_and_save_data(stat_name, x, e, var):
    # Plot and save mean and variance for a statistic

    # Plot variance
    plt.figure()
    plt.plot(np.log10(x),var / e)
    #plt.legend(("Sample variance divided by mean for %s" % stat_name,))
    plt.legend(("%s" % stat_name,))
    plt.xlabel("log(track size)")
    plt.ylabel("var(T) / E(T)")
    plt.savefig(FOLDER + "/%s_variance.png" % stat_name)
    np.savetxt(FOLDER + "/%s_variance.txt" % stat_name, [x, var])

    # Plot mean
    plt.figure()
    plt.plot(np.log10(x), e)
    #plt.legend(("Mean for %s" % stat_name,))
    plt.legend(("%s" % stat_name,))
    plt.xlabel("log(track size)")
    plt.ylabel("E(T)")
    plt.savefig(FOLDER + "/%s_mean.png" % stat_name)
    np.savetxt(FOLDER + "/%s_mean.txt" % stat_name, [x, e])




def simulate_stat_names_groupk(stat_names):
    ks = np.power(2, np.linspace(0,4,100))
    ks = np.linspace(2,16,100)
    p = 1e-4
    Ess = []
    varss = []
    Dss = []
    Fss = []
    for stat_name in stat_names:
        func = statistic_functions[stat_name]
        Es,vars_ = simulate_stat_by_k(func, ks, 10000, p)
        Ess.append(Es)
        varss.append(vars_)
        Dss.append(np.diff(Es))
        Fss.append(np.sqrt(vars_[1:])/(np.diff(Es)/np.diff(ks)))

    plt.figure()
    plt.plot(ks[1:],np.transpose(Fss))
    plt.legend(group)
    plt.xlabel("k")
    plt.ylabel("Std(T)/E(T)/(dE(t)/dk)")
    plt.show()
    plt.savefig("diffkplot.png")
    return ks[1:],np.array(Fss).transpose()

def simulate_stat_names_group(stat_names, k = 2.0, n_sim = 500):

    #stat_names = ["cosine", "driver-kroeber", "eyraud", "forbes1", "forbes2", "gilbert-wells", "kulczynski2","mcconnaughey", \
    #          "mountford", "pearson and heron-II","sample", "simpson", "sokal-sneath4", "tarantula", "tarwid", "YULEw", "t6"]

    Ess = []
    varss = []
    ms = []
    xs = np.power(10, np.linspace(-5,-2,150))
    ks = np.linspace(1,5,50)
    for stat_name in stat_names:
        print "Statistic: " + stat_name
        func = statistic_functions[stat_name]
        #Es,vars_ = simulate_stat_by_k(func, ks, 500)
        Es,vars_ = simulate_stat(func, xs, n_sim , k)
        ms.append(np.mean(Es))
        Ess.append(Es)
        varss.append(vars_)

        #plot_and_save_data(stat_name,xs,Es,vars_)


    """
    ones = []
    halfs =[]
    zeros = []
    mones = []
    for i, stat in enumerate(stat_names):
        if ms[i]>0.75:
            ones.append(stat)
        elif ms[i]>0.25:
            halfs.append(stat)
        elif ms[i]>-0.25:
            zeros.append(stat)
        else:
            mones.append(stat)
    print ones
    print halfs
    print zeros
    print mones
    exit(0)
    """
    #xs = ks

    # Hacky solution to get the stats with lowest varss only
    """
    varss = [[stat_names[i], varss[i]] for i in range(0, len(stat_names))]
    varss = sorted(varss, key = lambda t: np.mean(t[1]), reverse=False)
    varss = varss[0:6]
    print varss
    """
    #exit(0)
    #stat_names = [v[0] for v in varss]
    #varss = [v[1] for v in varss]
    #print stat_names
    #print varss
    print Ess

    #Ess = np.transpose(np.array(Ess))
    #varss = np.transpose(np.array(varss))


    # Plot mean
    ax = []
    for i in range(0, len(stat_names)):

        #plt.plot(xs,Ess)
        #plt.plot(xs,Ess/np.sqrt(varss))
        pl, = plt.plot(np.log10(xs), Ess[i], color=colors[k_counter])
        plt.plot(np.log10(xs), Ess[i] - 3*np.sqrt(varss[i]), color=colors[k_counter], ls='dashed')
        plt.plot(np.log10(xs), Ess[i] + 3*np.sqrt(varss[i]), color=colors[k_counter], ls='dashed')
        #pl, = plt.plot(xs, Ess[i], color=colors[i])
        #plt.plot(xs, Ess[i] - 5*varss[i], color=colors[i], ls='dashed')
        #plt.plot(xs, Ess[i] + 5*varss[i], color=colors[i], ls='dashed')

        ax.append(pl)



    plt.ylabel("E(T)")
    plt.xlabel("log(track size)")
    plt.xlabel("log10(1/track size)")
    #plt.xlabel("Track effect = p1/p0")

    #np.savetxt(FOLDER + "/all_means_with_stds_raw_data.txt", (Ess, varss, np.log10(xs), stat_names) )

    return ax

    """
    # Plot mean
    plt.figure()
    #plt.plot(xs,Ess)
    #plt.plot(xs,Ess/np.sqrt(varss))
    plt.plot(np.log10(xs), Ess)
    plt.legend(stat_names)
    plt.ylabel("E(T)")
    plt.xlabel("log(track size)")
    plt.savefig(FOLDER + "/all_means.png")

    # Plot variance
    plt.figure()
    #plt.plot(xs,Ess)
    #plt.plot(xs,Ess/np.sqrt(varss))
    plt.plot(np.log10(xs),varss / np.median(Ess, 0))
    plt.legend(stat_names)
    plt.ylabel("var(T) / E(T)")
    plt.xlabel("log(track size)")
    plt.savefig(FOLDER + "/all_variances.png")
"""

if __name__=="__main__":

    stat_names_in_groups = [['forbes1', 'sample', 'tarantula'],
    ['sokal-sneath4'],
    ['cosine', 'driver-kroeber', 'eyraud', 'forbes2', 'gilbert-wells', 'kulczynski2', 'mountford', 'pearson and heron-II', 'simpson', 'tarwid', 'YULEw', 't6', "YULEQ"],
    ['mcconnaughey']
    ]
    """
    #stat_names_in_groups = [["cosine", "driver-kroeber", "eyraud", "forbes1", "forbes2", "gilbert-wells", "kulczynski2","mcconnaughey", \
              "mountford", "pearson and heron-II","sample", "simpson", "sokal-sneath4", "tarantula", "tarwid", "YULEw", "YULEQ"],
                ["cosine", "forbes1", "mountford"]]

    # Very low variance on neutral tracks
    stat_names_in_groups = [['sokal-sneath4', 'kulczynski2', 'driver-kroeber', 'forbes2', 'simpson', 'mcconnaughey']]

    stat_names_in_groups = [["cosine", "driver-kroeber", "eyraud", "forbes1", "forbes2", "gilbert-wells", "kulczynski2","mcconnaughey",\
                             "mountford", "pearson and heron-II","sample", "simpson", "sokal-sneath4", "tarantula", "tarwid", "YULEw", "YULEQ"]]

    """

    stat_names_in_groups = [["onlya"]]#, "aoverbnorm", "forbes2"]]

    stat_names_in_groups = [statistic_functions.keys()[0:2]]
    stat_names_in_groups = [["cosine", "forbes1", "gilbert-wells", "pearson and heron-II", "YULEQ", "YULEw", "tarwid"]]

    stat_names_in_groups = [["jaccard", "forbes1"]]#, "jaccard"]]
    stat_names_in_groups = [["jaccard"]]#, "jaccard"]]
    stat_names_in_groups = [["tetrachoric"]]#, "jaccard"]]
    #stat_names_in_groups = [["forbes1"]]#, "jaccard"]]
    stat_names_in_groups = [["forbes1"], ["tetrachoric"]]#, "jaccard"]]



    FOLDER = "mean_and_vars"
    k_counter = 0 # Global

    plt.figure()
    for group in stat_names_in_groups:
        legends = []
        axes = []
        #for k in [1, 2, 4, 8]:
        for k in [1, 4]:
            ax = simulate_stat_names_group(group, k=k, n_sim = 500  )
            k_counter += 1
            axes += ax

            legends += [stat_name + ", k=%d" % k for stat_name in group]
        print legends
        print axes
        plt.legend(axes, legends)

        plt.savefig(FOLDER + "/all_means_with_stds.png")
        plt.show()
        #ks,Fss = simulate_stat_names_groupk(group)
