from __future__ import division, print_function

import math as mt

import numpy as np

import Output


# Calculates nCr (total no. of combinations)
def ncr(n, r):
    f = mt.factorial
    return f(n) // (f(r) * f(n - r))


# Calculates nC0 + nC1 + ... + nCk
def sum_ncr(n, k):
    s = 0
    for r in range(k):
        s += ncr(n, r)
    return s


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


# Calculates the density matrix for sub-lattice A
def density_matrix_a(label, e_vec, nos, nol_a, nop):
    dim_a = int(sum_ncr(nol_a, nop + 1))
    density_mat_a = np.zeros(shape=(dim_a, dim_a), dtype=complex)

    for i in range(nos):
        for j in range(nos):
            if label[i][1] == label[j][1] and label[i][2] == label[j][2]:
                m = int(label[i][0] + sum_ncr(nol_a, label[i][1]) - 1)
                n = int(label[j][0] + sum_ncr(nol_a, label[j][1]) - 1)
                density_mat_a[m][n] += np.vdot(e_vec[j], e_vec[i])

    # Calculates trace & trace of square of density matrix A
    den_trace_a = np.trace(density_mat_a.real)
    # den_trace_a2 = np.trace(np.linalg.matrix_power(density_mat_a, 2))

    # Error checking to make sure trace of DM remains ~1.0
    if mt.fabs(den_trace_a - 1.0) > 1.0e-5:
        Output.warning('Trace of density matrix A is not 1, Trace=', den_trace_a)

    return density_mat_a


# Calculates the density matrix, its trace and the trace of the square of the density matrix for sub-lattice B
def density_matrix_b(label, e_vec, nos, nol_b, nop):
    dim_b = sum_ncr(nol_b, nop + 1)
    density_mat_b = np.zeros(shape=(dim_b, dim_b), dtype=complex)

    for i in range(nos):
        for j in range(nos):
            if label[i][1] == label[j][1] and label[i][0] == label[j][0]:
                m = int(label[i][2] + sum_ncr(nol_b, (nop - label[i][1])) - 1)
                n = int(label[j][2] + sum_ncr(nol_b, (nop - label[j][1])) - 1)
                density_mat_b[m][n] += np.vdot(e_vec[j], e_vec[i])

    # Calculate trace & trace square of density matrix B
    den_trace_b = np.trace(density_mat_b.real)
    den_trace_b2 = np.trace(np.linalg.matrix_power(density_mat_b, 2))

    # Error checking to make sure trace of DM remains ~1.0
    if mt.fabs(den_trace_b - 1.0) > 1.0e-3:
        Output.warning('Trace of density matrix B is not 1, Trace=', den_trace_b)

    return density_mat_b, den_trace_b2
