from __future__ import division, print_function

import math as mt

import numpy as np
import scipy.linalg as la


# Calculates nCr (total no. of combinations), uses floor/integer division from Py3
def ncr(n, r):
    f = mt.factorial
    return f(n) // (f(r) * f(n - r))


# Relabels states
def relabel(e_states, nop, link_pos, nol_b):
    x = np.zeros(shape=(2, nop + 1), dtype=np.int32)
    y, dump = [], []
    for state in e_states:
        comm, temp = [], []
        n = 0

        for j in state:
            if j <= link_pos:
                comm.append(j)
                n += 1

        x[1][n] += 1

        if comm not in dump:
            x[0][n] += 1
            dump.append(comm)

        temp += [x[0][n], n, x[1][n]]
        y.append(temp)

        if x[1][n] == ncr(nol_b, nop - n):
            x[1][n] = 0
    return np.array(y)


# Returns an initial state by placing eigenvectors from A in a zero matrix
def init_state(eigenvectors_a, relabelled_states, nos, nop, state_num):
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
