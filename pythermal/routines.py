# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Time evolving hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group
# St. Stephen's Centre for Theoretical Physics, New Delhi
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import division, print_function, absolute_import

import ctypes
import itertools as it
import math as mt
import multiprocessing as mp
import warnings

import numpy as np
from scipy.linalg import eig, block_diag
from tqdm import tqdm

try:
    from builtins import range
except ImportError:
    from __builtin__ import range

__all__ = ['position_states', '_hamiltonian', 'distribute',
           'hamiltonian_parallel', 'eigen', 'ncr', 'sum_ncr', 'relabel',
           'density_matrix_a', 'rho_b_pbasis',
           'h_block_diagonal', 'transformation', 'naive_thermal']


def position_states(lat, nop, del_pos=None):
    """
    Returns position states for a given lattice & particles after deleting
    sites
    :param lat: Array of lattice sites
    :param nop: Nop of particles in lattice
    :param del_pos: Lattice sites to delete
    :return: Positions states
    :return: Total no. of states
    """
    if del_pos is None:
        pos_states = np.array(list(it.combinations(lat, nop)), dtype=np.int32)
    else:
        lat_del = np.delete(lat, del_pos - 1)
        pos_states = np.array(list(it.combinations(lat_del, nop)),
                              dtype=np.int32)

    return pos_states, len(pos_states)


# Core hamiltonian function
def _hamiltonian(start, stop, nos, ndims, nop, pos_states):
    for j in tqdm(range(start, stop)):
        for k in range(nos):
            # Find common elements, sum and no.
            c = np.intersect1d(pos_states[j], pos_states[k])
            c_sum = np.sum(c, dtype=int)
            c_size = np.size(c)

            j_sum = np.sum(pos_states[j], dtype=int)
            k_sum = np.sum(pos_states[k], dtype=int)

            if c_size == nop - 1:
                # Differ by one element
                if abs(j_sum - k_sum) == ndims:
                    # Differ by dimension
                    ham[j, k] = 1
                elif (k_sum - j_sum) == 1 and not (j_sum - c_sum) % ndims == 0:
                    # Right/Left edge
                    ham[j, k] = 1
                elif (j_sum - k_sum) == 1 and not (j_sum - c_sum) % ndims == 1:
                    # Right/Left edge
                    ham[j, k] = 1
                else:
                    ham[j, k] = 0
            else:
                ham[j, k] = 0


def distribute(n_items, n_processes, i):
    """
    Distributes items among processes.
    :param n_items: Total no. of items
    :param n_processes: Total no. of processes
    :param i: Process no.
    :return: Start & Stop point index in no. of items
    """
    items_per_process = n_items // n_processes  # Integer division
    start = i * items_per_process

    if i == n_processes - 1:
        # For last process, appends all remaining items to last core
        stop = n_items
    else:
        stop = items_per_process * (i + 1)

    return start, stop


def hamiltonian_parallel(lattice, ndims, nop):
    """
    Wrapper for _hamiltonian. Creates multiple processes, each of which
    process and forms a part of the hamiltonian array which is placed in
    shared memory.
    :param lattice: Lattice used
    :param ndims: Dimensionality of lattice
    :param nop: No. of particles
    :return: Hamiltonian matrix
    """
    pos_states, nos = position_states(lattice, nop)

    # Creates a multiprocessing Array and reshapes it for the hamiltonian
    # The Array is declared globally within this function
    # The Array is a part of shared memory for various processes
    # Has been known to cause memory leaks [Use with caution]
    global ham
    ham_base = mp.Array(ctypes.c_int8, nos * nos)
    ham = np.ctypeslib.as_array(ham_base.get_obj())
    ham = ham.reshape(nos, nos)

    # No. of processes to create for parallel processing
    n_processes = mp.cpu_count()
    process_list = []

    for i in range(n_processes):
        start, stop = distribute(nos, n_processes, i)
        args = (start, stop, nos, ndims, nop, pos_states)
        process = mp.Process(target=_hamiltonian, args=args)
        process_list.append(process)
        process.start()

    for processes in process_list:
        processes.join()

    return ham


def eigen(h):
    """
    Calculates eigenvectors and eigenvalues used Pade algorithm
    (link to OpenBLAS Fortran libraries for parallel processing)
    :param h: Hamiltonian matrix
    :return: Real array of eigenvalues
    :return: Complex array of eigenvectors
    """
    eigenvalues, eigenvectors = eig(h, check_finite=False)

    # Select only real part of eigenvalues
    eigenvalues = eigenvalues.real

    # Sort eigenvalues and eigenvectors by ascending eigenvalue
    index = eigenvalues.argsort()
    eigenvectors = eigenvectors[:, index]
    eigenvalues = eigenvalues[index]

    return eigenvalues, eigenvectors


def ncr(n, r):
    """
    :param n: Total no. of items
    :param r: No. of items chosen
    :return: Total no. of combinations
    """
    f = mt.factorial
    return f(n) // (f(r) * f(n - r))


def sum_ncr(n, k):
    """
    Calculates nC0 + nC1 + ... + nCr
    :param n: Total no. of items
    :param k: No. of items chosen
    :return: Sum of combinations
    """
    return sum(ncr(n, r) for r in range(k))


def relabel(e_states, nop, nol_b, lat_a=None):
    """
    Relabels states.
    :param lat_a: Sub-lattice B
    :param e_states: Eigenstates
    :param nop: No. of particles
    :param nol_b: No. of lattice sites in B
    :return: Array of relabelled states
    """
    x = np.zeros(shape=(2, nop + 1), dtype=np.int32)
    relabelled_states, dump = [], []

    for state in e_states:
        temp = []

        comm = [k for k in state if k in lat_a]
        n = len(comm)

        x[1][n] += 1

        if comm not in dump:
            x[0][n] += 1
            dump.append(comm)

        temp += [x[0][n], n, x[1][n]]
        relabelled_states.append(temp)

        if x[1][n] == ncr(nol_b, nop - n):
            x[1][n] = 0

    return np.array(relabelled_states)


def density_matrix_a(label, e_vec, nos, nol_a, nop):
    """
    Calculates density matrix for sub-lattice B
    :param label: Relabelled states
    :param e_vec: Eigenvectors
    :param nos: No. of states
    :param nol_a: No. of lattice sites in A
    :param nop: No. of particles
    :return: Density matrix of sub-lattice A
    """
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

    # Error checking to make sure trace of DM remains ~1.0
    if mt.fabs(den_trace_a - 1.0) > 1.0e-1:
        warnings.warn('Trace of density matrix A not 1, Trace=', den_trace_a)

    return density_mat_a


def rho_b_pbasis(label, e_vec, nos, nol_b, nop):
    """
    Calculates density matrix for sub-lattice B. Error checks density matrix
    whose trace should be 1.0. Warning raised if trace differs by >0.1.
    :param label: Relabelled states
    :param e_vec: Eigenvectors
    :param nos: No. of states
    :param nol_b: No. of lattice sites in B
    :param nop: No. of particles
    :return: Density matrix of sub-lattice B
    """
    dim_b = sum_ncr(nol_b, nop + 1)
    rho_b = np.zeros(shape=(dim_b, dim_b), dtype=complex, order='F')

    for i in tqdm(range(nos)):
        for j in range(nos):
            if label[i, 1] == label[j, 1] and label[i, 0] == label[j, 0]:
                m = int(label[i, 2] + sum_ncr(nol_b, (nop - label[i, 1])) - 1)
                n = int(label[j, 2] + sum_ncr(nol_b, (nop - label[j, 1])) - 1)

                rho_b[m, n] += np.vdot(e_vec[j], e_vec[i])

    tr_rho = np.trace(rho_b.real, dtype=float)
    if mt.fabs(tr_rho - 1.0) > 1.0e-1:
        print('Trace of density matrix B not 1, Trace=', tr_rho)

    return rho_b


def h_block_diagonal(lat_b, n_dim, nop):
    bd = []
    for i in range(nop + 1):
        bd.append(hamiltonian_parallel(lat_b, n_dim, i))
    return block_diag(*bd)


def transformation(rho_pbasis, e_vecs_bd):
    """
    Transforms rho in position basis to rho in energy basis.
    :TODO Make faster
    :param rho_pbasis: rho in position basis
    :param e_vecs_bd: eigenvectors of block diagonal hamiltonian
    :return: rho in energy basis
    """
    rho_ebasis = np.zeros_like(rho_pbasis, dtype=complex)

    for i in tqdm(range(rho_pbasis.shape[0])):
        for j in range(rho_pbasis.shape[1]):
            rho_ebasis[i, j] = \
                np.vdot(e_vecs_bd[:, i], np.dot(rho_pbasis, e_vecs_bd[:, j]))

    if not np.allclose(np.transpose(np.conjugate(rho_ebasis)), rho_ebasis):
        print("Transformation matrix is not symmetric")

    return rho_ebasis


def naive_thermal(rho):
    """
    Compare maximum diagonal and off diagonal terms in density matrix.
    CAUTION: Destroys original density matrix.
    :param rho:
    :return: Max diagonal element
    :return: Max off-diagonal element
    """
    max_diag = rho.diagonal().max()
    print("Max diagonal = {}".format(max_diag))

    rho_copy = np.copy(rho)
    np.fill_diagonal(rho_copy, -np.inf)
    max_offdiag = rho_copy.max()
    print("Max off-diagonal = {}".format(max_offdiag))

    print("Ratio (Diagonal/Off-diagonal) = {}".format(max_diag / max_offdiag))
    return max_diag, max_offdiag
