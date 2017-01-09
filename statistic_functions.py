import numpy as np
from collections import OrderedDict
#from math import sqrt, log, cos, pi
from numpy import sqrt, log, cos, pi, maximum, minimum, log

def sigma(a, b, c, d, n):
    return maximum(a, b) + maximum(c,d) + maximum(a,c) + maximum(b,d)

def sigmam(a, b, c, d, n):
    return maximum(a + c, b + d) + maximum(a + b, c + d)

def ro(a,b,c,d,n):
    return (a*d - b*c) / sqrt((a+b)*(a+c)*(b+d)*(c+d))

def chi_square(a,b,c,d,n):
    return (n*(a*d - b*c)**2) / ((a+b)*(a+c)*(b+d)*(c+d))


statistic_functions = {
                        "sample" : lambda a, b, c, d, n: abs(a * (c+d) / (c * (a+b))),
                        "tarantula" : lambda a, b, c, d, n:  a * (c+d) / (c * (a+b)),
                        "eyraud" : lambda a, b, c, d, n: (n**2 * (n*a - (a + b)*(a + c))) / ((a+b)*(a+c)*(b+d)*(c+d)),
                        "pierce" : lambda a, b, c, d, n: (a*b + b*c) / (a*b + 2*b*c + c*d),
                        "baroni urbani and buser I" : lambda a, b, c, d, n: (sqrt(a * d) + a) / (sqrt(a * d) + a + b+ c),
                        "baroni urbani and buser II" : lambda a, b, c, d, n: (sqrt(a * d) + a - (b + c)) / (sqrt(a * d) + a + b+ c),
                        "anderberg": lambda a,b,c,d,n: (sigma(a,b,c,d,n) - sigmam(a,b,c,d,n)) / (2*n) ,
                        "goodman and kruskal": lambda a,b,c,d,n: (sigma(a,b,c,d,n) - sigmam(a,b,c,d,n)) / (2*n - sigmam(a,b,c,d,n)) ,
                        "michael": lambda a,b,c,d,n: (4 * (a*d - b*c)) / ((a+d)**2 + (b+c)**2),
                        "hamann": lambda a,b,c,d,n: ((a+d) - (b+c)) / (a+b+c+d) ,
                        "disperson": lambda a,b,c,d,n: (a*d - b*c) / (a + b+ c+ d)**2 ,
                        "tanimoto": lambda a,b,c,d,n: a / ((a + b) + (a + c) - a) ,
                        "kulcynski-I": lambda a,b,c,d,n: a / (b+c)  ,
                        "YULEw": lambda a,b,c,d,n: (sqrt(a*d) - sqrt(b*c)) / (sqrt(a*d) + sqrt(b*c)) ,
                        "YULEQ": lambda a,b,c,d,n: (a*d - b*c) / (a*d + b*c) ,
                        "ochiai-II": lambda a,b,c,d,n: (a*d) / sqrt((a+b) * (a+c) * (b+d) * (c+d))  ,
                        "stiles": lambda a,b,c,d,n: np.log10( (n * (abs(a*d - b*c) - n/2)**2) / ((a+b)*(a+c)*(b+d)*(c+d))) ,
                        "cole": lambda a,b,c,d,n: (sqrt(2) * (a*d - b*c)) / (sqrt((a*d-b*c)**2 - (a+b)*(a+c)*(b+d)*(c+d))),
                        "sokal and sneath-V": lambda a,b,c,d,n: (a*d) / sqrt((a+b)*(a+c)*(b+d)*(c+d)),
                        "sokal and sneath-III": lambda a,b,c,d,n: (a+d) / (b+c) ,
                        "pearson and heron-II": lambda a,b,c,d,n: cos( (pi * sqrt(b*c)) / (sqrt(a*d) + sqrt(b*c))) ,
                        "pearson and heron-I": lambda a,b,c,d,n: (a*d - b*c) / sqrt((a+b)*(a+c)*(b+d)*(c+d)),
                        "pearson-III": lambda a,b,c,d,n: sqrt(ro(a,b,c,d,n) / n + ro(a,b,c,d,n)) ,
                        "pearson-II": lambda a,b,c,d,n: sqrt(chi_square(a, b, c, d, n) / (n+chi_square(a, b, c, d, n)))  ,
                        "pearson-I": lambda a,b,c,d,n: chi_square(a, b, c, d, n) ,
                        "gower": lambda a,b,c,d,n: (a+d) / sqrt((a+b)*(a+c)*(b+d)*(c+d)),
                        "jaccard": lambda a,b,c,d,n:a/(a+b+c),
                        "log jaccard": lambda a,b,c,d,n:log(a/(a+b+c)),
                        "dice": lambda a,b,c,d,n:2*a/(2*a+b+c),
                        "czekanowski": lambda a,b,c,d,n:2*a/(2*a+b+c),
                        "bw-jaccard": lambda a,b,c,d,n:3*a/(3*a+b+c),
                        "nei-li": lambda a,b,c,d,n:2*a/(a+b+a+c),
                        "sokal-sneath1": lambda a,b,c,d,n:a/(a+2*b+2*c),
                        "sokal-michener": lambda a,b,c,d,n: (a+d)/(a+b+c+d),
                        "sokal-sneath2": lambda a,b,c,d,n:(2*(a+d))/(2*a+b+c+2*d),
                        "roger-tanimoto": lambda a,b,c,d,n:(a+d)/(a+2*(b+c)+d),
                        "faith": lambda a,b,c,d,n:(a+0.5*d)/(a+b+c+d),
                        "gower-legendre": lambda a,b,c,d,n:(a+d)/(a+0.5*(b+c)+d),
                        "russel-rao": lambda a,b,c,d,n:a/(a+b+c+d),
                        "hamming": lambda a,b,c,d,n:b+c,
                        "cosine": lambda a,b,c,d,n:n*a/((a+b)*(a+c)), # Should not have n factor
                        "cosine_original": lambda a,b,c,d,n:a/((a+b)*(a+c)),
                        "gilbert-wells": lambda a,b,c,d,n:log(a)-log(n)-log((a+b)/n)-log((a+c)/n),
                        "ochiai1": lambda a,b,c,d,n:a/sqrt((a+b)*(a+c)),
                        "forbes1": lambda a,b,c,d,n:n*a/((a+b)*(a+c)),
                        "log forbes1": lambda a,b,c,d,n: log(n*a/((a+b)*(a+c))),
                        "t5": lambda a,b,c,d,n:n*a/((a+b)*(a+c)),
                        "fossum": lambda a,b,c,d,n:n*(a-0.5)**2/((a+b)*(a+c)),
                        "sorgenfrei": lambda a,b,c,d,n:a**2/((a+b)*(a+c)),
                        "mountford": lambda a,b,c,d,n:n*a/(0.5*(a*b+a*c)+b*c), # Should not have n factor
                        "mountford_original": lambda a,b,c,d,n:a/(0.5*(a*b+a*c)+b*c),
                        "otsuka": lambda a,b,c,d,n:a/sqrt(((a+b)*(a+c))),
                        "mcconnaughey": lambda a,b,c,d,n:(a**2-b*c)/((a+b)*(a+c)),
                        "tarwid": lambda a,b,c,d,n:(n*a-(a+b)*(a+c))/(n*a+(a+b)*(a+c)),
                        "kulczynski2": lambda a,b,c,d,n:(a/2*(2*a+b+c))/((a+b)*(a+c)),
                        "driver-kroeber": lambda a,b,c,d,n:a/2*(1/(a+b)+1/(a+c)),
                        "johnson": lambda a,b,c,d,n:a/(a+b)+a/(a+c),
                        "dennis": lambda a,b,c,d,n:(a*d-b*c)/sqrt(n*(a+b)*(a+c)),
                        "simpson": lambda a,b,c,d,n:a/minimum(a+b,a+c),
                        "braun-banquet": lambda a,b,c,d,n:a/maximum(a+b,a+c),
                        "fager-mcgowan": lambda a,b,c,d,n:a/sqrt((a+b)*(a+c))-maximum(a+b,a+c)/2,
                        "forbes2": lambda a,b,c,d,n:(n*a-(a+b)*(a+c))/(n*minimum(a+b,a+c)-(a+b)*(a+c)),
                        "sokal-sneath4": lambda a,b,c,d,n:(a/(a+b)+a/(a+c)+d/(b+d)+d/(c+d))/4,
                        "t6": lambda a,b,c,d,n: (a - (a+c)*(a+b)/n) / ((a+c)*(1 - (a+b)/n)),
                        "t6v2": lambda a,b,c,d,n: (a - (a+b)*(a+c)/n) / sqrt(a*(1-a/(a+c)) + c*(1-b/(d+c))),
                        "aoverb": lambda a,b,c,d,n: (a - b) / (a+b),
                        "aoverbnorm": lambda a,b,c,d,n :  (a / sqrt(a * (1-a/(a+c)))) / (b/ sqrt(b*(1-b/(n-(a+c)))) ),
                        "onlya": lambda a,b,c,d,n: a / sqrt(a * (1-a/(a+c))),
                        "tetrachoric": lambda a,b,c,d,n: cos ((pi/2)/(1 + sqrt((b*c)/(a*d))))
                        #"aoverbnorm": lambda a,b,c,d,n : 1.0* (a / np.sqrt((a+c)*(a/n) * (1-a/n))) / (b/ np.sqrt((1-a-c) * (b/n)*(1-b/n)) )
                        #"aoverbnorm": lambda a,b,c,d,n : ((a+c) - (1.0-a-c)) * (a / np.sqrt((a+c)*(a/n) * (1-a/n))) / (b/ np.sqrt((1-a-c) * (b/n)*(1-b/n)) )
                        #"aoverbnorm": lambda a,b,c,d,n : ((a+c) / (1-a-c)) * (a / ((a+c)*(a/n) * (1-a/n))) / (b/((1-a-c) * (b/n)*(1-b/n)) )
}

statistic_functions = OrderedDict(sorted(statistic_functions.items(), key=lambda t: t[0]))
#"""
#                        "jaccard": lambda a,b,c,d,n:a/(a+b+c),
#                        "dice": lambda a,b,c,d,n:2*a/(2*a+b+c),
#                        "czekanowski": lambda a,b,c,d,n:2*a/(2*a+b+c),
#                        "bw-jaccard": lambda a,b,c,d,n:3*a/(3*a+b+c),
#                        "nei-li": lambda a,b,c,d,n:2*a/(a+b+a+c),
#                        "sokal-sneath1": lambda a,b,c,d,n:a/(a+2*b+2*c),
#                        "sokal-michener": lambda a,b,c,d,n: (a+d)/(a+b+c+d),
#                        "sokal-sneath2": lambda a,b,c,d,n:(2*(a+d))/(2*a+b+c+2*d),
#"""


if __name__ == "__main__":

    a = 2 * np.ones(5);
    b = 1 * np.ones(5);
    c = 1 * np.ones(5);
    d = 4 * np.ones(5);
    n = 10

    for func in statistic_functions:
        print "Function %s: %s " % (func, str(statistic_functions[func](a, b, c, d, n)))
