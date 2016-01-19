# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Time evolving hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import division, print_function

import math as mt

import numpy as np
import scipy.linalg as la


def ncr(n, r):
    """
    :param n: Total no. of items
    :param r: No. of items chosen
    :return: Total no. of combinations

    """
    f = mt.factorial
    return f(n) // (f(r) * f(n - r))


def relabel(e_states, nop, nol_b, link_pos=None, lat_b=None):
    """
    Relabels states
    :param lat_b: Sub-lattice B
    :return:
    :param e_states: Eigenstates
    :param nop: No. of particles
    :param link_pos: Site linking arrays
    :param nol_b: No. of lattice sites in B
    :return:

    """
    x = np.zeros(shape=(2, nop + 1), dtype=np.int32)
    y, dump = [], []

    for state in e_states:
        comm, temp = [], []

        if lat_b is None:
            n, comm = num_a(comm, state, link_pos)

        elif link_pos is None:
            n, comm = num_a_manual(comm, state, lat_b)

        else:
            raise Exception('Specify linking position or define sub-lattice B')

        x[1][n] += 1

        if comm not in dump:
            x[0][n] += 1
            dump.append(comm)

        temp += [x[0][n], n, x[1][n]]
        y.append(temp)

        if x[1][n] == ncr(nol_b, nop - n):
            x[1][n] = 0

    return np.array(y)


def num_a(comm, state, link_pos):
    n = 0
    for j in state:
        if j <= link_pos:
            comm.append(j)
            n += 1
    return n, comm


def num_a_manual(comm, state, lat_b):
    n = 0
    for j in state:
        if j in lat_b:
            comm.append(j)
            n += 1

    return n, comm


def init_state(eigenvectors_a, relabelled_states, nos, nop, state_num):
    """
    Returns a normalized initial state by placing eigenvectors from A in a zero matrix
    :param eigenvectors_a: Eigenvector of sub lattice A
    :param relabelled_states: Array of relabelled states
    :param nos: No. of states
    :param nop: No. of particles
    :param state_num: Initial eigenvector chosen
    :return: Normalized initial state

    """
    if state_num > len(eigenvectors_a):
        raise ValueError('Eigenvector not in range. Range(0 - {})'.format(len(eigenvectors_a)))

    psi_initial = np.zeros(nos, dtype=np.complex)
    j = 0

    # Iterates over second column of RS
    for idx, val in enumerate(relabelled_states[:, 1]):
        if val == nop:
            psi_initial[idx] = eigenvectors_a[state_num, j]
            j += 1

    # Normalizes initial state
    psi_initial /= la.norm(psi_initial)

    return psi_initial
