##
import numpy as np
# COPULA IMPORTS
import context
from starvine.bvcopula.copula import *
from bvcopula import bv_plot
# Plotting
import matplotlib.pyplot as plt
# import seaborn as sns


def plot_hdist():
    n = 100
    theta = 0.7
    plt.figure(1)
    ax = plt.subplot(111)

    u = np.linspace(1e-9, 1 - 1e-9, n)
    for v in np.linspace(1e-2, 1 - 1e-2, 10):
        conditioning_v = np.ones(n) * v
        c = t_copula.StudentTCopula()
        h_v = c.h(u, conditioning_v, *[theta, 10])
        label_str = r"$v$: %.2f, $\theta$: %.1e" % (v, theta)
        ax.plot(u, h_v, label=label_str)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1., 0.5), fontsize=14)
    plt.xlabel(r"$u$", fontsize=16)
    plt.ylabel(r"$h(u;v,\theta)$", fontsize=16)
    plt.savefig("t_h_dist.png")


def plot_hinv_dist():
    n = 100
    theta = 0.7
    plt.figure(2)
    ax = plt.subplot(111)

    u = np.linspace(1e-9, 1 - 1e-9, n)
    for v in np.linspace(1e-2, 1 - 1e-2, 10):
        conditioning_v = np.ones(n) * v
        c = t_copula.StudentTCopula()
        h_v = c.hinv(u, conditioning_v, *[theta, 10])
        label_str = r"$v$: %.2f, $\theta$: %.1e" % (v, theta)
        ax.plot(u, h_v, label=label_str)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1., 0.5), fontsize=14)
    plt.xlabel(r"$u$", fontsize=16)
    plt.ylabel(r"$h^{-1}(u;v,\theta)$", fontsize=16)
    plt.savefig("t_hinv_dist.png")

if __name__ == "__main__":
    plot_hdist()
    plot_hinv_dist()
