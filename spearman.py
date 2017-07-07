#########################################################################
# File Name: spearman.py
# Author: Haoyue Shi
# mail: freda.haoyue.shi@gmail.com
# Created Time: 2017-06-18 20:04
# Description: This script implements spearman rank correlation test.
#########################################################################
from pprint import pprint


def rank(x):
    sa = []
    ret = [0 for i in xrange(len(x))]
    for i in range(len(x)):
        sa.append([i, x[i]])
    sa = sorted(sa, key=lambda x: x[1])
    i = 0
    while i < len(sa):
        j = i
        while j < len(sa) and sa[i][1] == sa[j][1]:
            j += 1
        ret[sa[i][0]] = i
        for k in range(i, j):
            ret[sa[k][0]] = float(i+j-1)/2
        i = j
    return ret


def clear(x, y):
    ret_x = []
    ret_y = []
    for i in range(len(x)):
        if x[i] != -1 and y[i] != -1:
            ret_x.append(x[i])
            ret_y.append(y[i])
    return ret_x, ret_y


def spearman(x, y):
    x, y = clear(x, y)
    x = rank(x)
    y = rank(y)
    n = len(x)
    ret = 0
    for i in range(n):
        ret += 6.0 * ((x[i]-y[i])**2)
    ret = 1 - ret / n / (n-1) / (n+1)
    return ret


def scws(file_name='../../data/SCWS/ratings.txt'):
    ratings = open(file_name).readlines()
    return map(lambda x: float(x.split()[len(x.split())-11]), ratings)
