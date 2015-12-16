import math as mt

import numpy as np

from Main import System


# Calculates nCr (total no. of combinations)
def ncr(n, r):
    f = mt.factorial
    return int(f(n) / (f(r) * f(n - r)))


# Calculates nC0 + nC1 + ... + nCk
def sum_ncr(n, k):
    s = 0
    for r in xrange(k):
        s += int(ncr(n, r))
    return s


# Relabels states
def relabel(e_states):
    s = System()
    
    x = np.zeros(shape=(2, s.nop + 1), dtype=np.int32)
    y, dump = [], []
    for state in e_states:
        comm, temp = [], []
        n = 0

        for j in state:
            if j <= s.link_pos:
                comm.append(j)
                n += 1

        x[1][n] += 1

        if comm not in dump:
            x[0][n] += 1
            dump.append(comm)

        temp += [x[0][n], n, x[1][n]]
        y.append(temp)

        if x[1][n] == ncr(s.nol_b, s.nop - n):
            x[1][n] = 0
    return np.array(y)


# Calculates the density matrix, its trace and the trace of the square of the density matrix for sub-lattice A
def denmatrix_a(label, e_vec, nos):
    s = System()

    dim_a = int(sum_ncr(s.nol_a, s.nop + 1))
    density_mat_a = np.zeros(shape=(dim_a, dim_a), dtype=complex)

    for i in xrange(nos):
        for j in xrange(nos):
            if label[i][1] == label[j][1] and label[i][2] == label[j][2]:
                m = int(label[i][0] + sum_ncr(s.nol_a, label[i][1]) - 1)
                n = int(label[j][0] + sum_ncr(s.nol_a, label[j][1]) - 1)
                density_mat_a[m][n] += np.vdot(e_vec[j], e_vec[i])

    den_trace_a = np.trace(density_mat_a.real)
    # den_trace_a2 = np.trace(np.linalg.matrix_power(density_mat_a, 2))
    # print "Trace a =", den_trace_a, "Trace a^2 =", den_trace_a2

    # Error checking to make sure trace of DM remains ~1.0
    if mt.fabs(den_trace_a - 1.0) > 1.0e-4:
        print "Trace of density matrix a is not 1"

    return density_mat_a


# Calculates the density matrix, its trace and the trace of the square of the density matrix for sub-lattice B
def denmatrix_b(label, e_vec, nos):
    s = System()

    dim_b = sum_ncr(s.nol_b, s.nop + 1)
    density_mat_b = np.zeros(shape=(dim_b, dim_b), dtype=complex)

    for i in xrange(nos):
        for j in xrange(nos):
            if label[i][1] == label[j][1] and label[i][0] == label[j][0]:
                m = int(label[i][2] + sum_ncr(s.nol_b, (s.nop - label[i][1])) - 1)
                n = int(label[j][2] + sum_ncr(s.nol_b, (s.nop - label[j][1])) - 1)
                density_mat_b[m][n] += np.vdot(e_vec[j], e_vec[i])

    den_trace_b = np.trace(density_mat_b.real)
    # den_trace_b2 = np.trace(np.linalg.matrix_power(density_mat_b, 2))
    # print "Trace b =", den_trace_b, "\t Trace b squared =", den_trace_b2

    # Error checking to make sure trace of DM remains ~1.0
    if mt.fabs(den_trace_b - 1.0) > 1.0e-4:
        print "Trace of density matrix B is not 1"

    return density_mat_b


def recursion_time(eigenvalues):
    min_time = np.amin(np.absolute(eigenvalues))

    return 1.0 / min_time


'''
def gcd(a, b):
    while b:
        a, b = b, a % b

    return a


def lcm(a, b):

    return a * b / gcd(a, b)


def lcm_call(*args):
    return reduce(lcm, args)


def recursion_time(rnd_off, eigenvalues):
    e_vals = np.zeros_like(eigenvalues, dtype=np.float32)

    for idx, val in enumerate(eigenvalues):
        e_vals[idx] = np.round(val, rnd_off)

    return lcm_call(*e_vals)
'''
