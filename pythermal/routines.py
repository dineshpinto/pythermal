# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Time evolving hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group
# St. Stephen's Centre for Theoretical Physics, New Delhi
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import division, print_function, absolute_import

import itertools as it
import math as mt
import multiprocessing as mp
import warnings

import numpy as np
import scipy.linalg as la
from tqdm import tqdm

try:
    from builtins import range
except ImportError:
    from __builtin__ import range

__all__ = ['eigenstates_lattice', 'hamiltonian_2d', 'distribute',
           'parallel_call_h', 'eig', 'ncr', 'sum_ncr', 'relabel',
           'initial_state', 'density_matrix_a', 'density_matrix_b',
           'trace_squared', 'vn_entropy_b', 'time_evolution', 'avg_particles']


def eigenstates_lattice(lat, nop, del_pos=None):
    """
    Returns eigenstates for a given lattice & particles after deleting sites
    :param lat: Array of lattice sites
    :param nop: Nop of particles in lattice
    :param del_pos: Lattice sites to delete
    :return: Eigenstates
    :return: Total no. of states

    """
    if del_pos is None:
        e_states = np.array(list(it.combinations(lat, nop)), dtype=np.int32)
    else:
        lat_del = np.delete(lat, del_pos - 1)
        e_states = np.array(list(it.combinations(lat_del, nop)),
                            dtype=np.int32)

    return e_states, len(e_states)


def hamiltonian_2d(start, stop, nos, ndims, nop, e_states, queue, h):
    """
    :param start: Start iterations at this point
    :param stop: Stop iterations at this point
    :param nos: No. of states
    :param ndims: Shape of lattice
    :param nop: No. of particles
    :param e_states: Array of eigenstates
    :param queue: Multiprocessing queue to store each processes' output
    :param h: Hamiltonian matrix

    """
    for j in tqdm(range(start, stop)):  # Start/Stop defined by distribute()

        for k in range(nos):  # k iterates over all possibilities

            c = np.intersect1d(e_states[j], e_states[k])
            # Sum of common elements
            c_sum = np.sum(c, dtype=np.int32)
            # No. of common elements
            c_size = np.size(c)

            j_sum = np.sum(e_states[j], dtype=np.int32)
            k_sum = np.sum(e_states[k], dtype=np.int32)

            if c_size == nop - 1:
                # Only one element differs

                if abs(j_sum - k_sum) == ndims:
                    # Element differs by dimension
                    h[j, k] = float(1)

                elif (k_sum - j_sum) == 1 and not (j_sum - c_sum) % ndims == 0:
                    # Right/Left edge
                    h[j, k] = float(1)

                elif (j_sum - k_sum) == 1 and not (j_sum - c_sum) % ndims == 1:
                    # Right/Left edge
                    h[j, k] = float(1)

                else:
                    continue

            else:
                continue

    queue.put(h)


def distribute(n_items, n_processes, i):
    """
    Distributes processes among processors
    :param n_items: Total no. of items
    :param n_processes: No. of processors/cores/threads
    :param i: Iterator over n_processes
    :return: Start point of ith process
    :return: Stop point of ith process

    """
    items_per_process = n_items // n_processes  # Integer division
    start = i * items_per_process

    # For last process, appends all remaining items to last core
    if i == n_processes - 1:
        stop = n_items
    else:
        stop = items_per_process * (i + 1)

    return start, stop


def parallel_call_h(e_states, nos, ndims, nop):
    """
    Multiple parallel calls to hamiltonian_2d.
    :param e_states: Array of eigenstates
    :param nos: No. of states
    :param ndims: Shape of lattice.
    :param nop: No. of particles
    :return: Hamiltonian matrix

    """
    process_list = []
    queue = mp.Queue()  # Setting up a queue to store each processes' output
    h = np.zeros(shape=(nos, nos), dtype=np.float32)
    # No. of processes to create for parallel processing
    n_processes = mp.cpu_count()

    for i in range(n_processes):
        start, stop = distribute(nos, n_processes, i)
        args = (start, stop, nos, ndims, nop, e_states, queue, h)

        process = mp.Process(target=hamiltonian_2d, args=args)
        process_list.append(process)  # Create list of processes
        process.start()

    for i in range(n_processes):  # Retrieves output from queue
        h += queue.get()

    while not queue.empty():  # Clear queue
        h += queue.get()

    for jobs in process_list:  # Joins processes together
        jobs.join()

    return h


def eig(h):
    """
    Calculates eigenvectors and eigenvalues used Pade algorithm
    (link to OpenBLAS Fortran libraries for parallel processing)
    :param h: Hamiltonian matrix
    :return: Real array of eigenvalues
    :return: Complex array of eigenvectors

    """
    eigenvalues, eigenvectors = la.eig(h, check_finite=False)

    # Sort eigenvalues and eigenvectors by ascending eigenvalue
    index = eigenvalues.argsort()
    eigenvectors = eigenvectors[:, index]
    eigenvalues = eigenvalues[index]

    return eigenvalues.real, eigenvectors


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


def relabel(e_states, nop, nol_b, link_pos=None, lat_b=None):
    """
    Relabels states.
    :param lat_b: Sub-lattice B
    :param e_states: Eigenstates
    :param nop: No. of particles
    :param link_pos: Site linking arrays
    :param nol_b: No. of lattice sites in B
    :return: Array of relabelled states

    """
    x = np.zeros(shape=(2, nop + 1), dtype=np.int32)
    relabelled_states, dump = [], []

    for state in e_states:
        temp = []

        if link_pos is not None:
            comm = [k for k in state if k <= link_pos]
        elif lat_b is not None:
            comm = [k for k in state if k not in lat_b]
        else:
            raise ValueError('Unspecified link point for lattices.')

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


def initial_state(e_vec, label, nos, nop, e_vec_num):
    """
    Returns a normalized initial state by placing eigenvectors from A in
    a matrix of zeros.

    :param e_vec: Eigenvector of sub lattice A
    :param label: Array of relabelled states
    :param nos: No. of states
    :param nop: No. of particles
    :param e_vec_num: Initial eigenvector chosen
    :return: Normalized initial state

    """
    if e_vec_num > len(e_vec):
        raise ValueError('Eigenvector not in range. Range(0 - {})'
                         .format(len(e_vec)))
    print(len(e_vec))
    psi_initial = np.zeros(nos, dtype=np.complex)

    j = 0
    for idx, val in enumerate(label[:, 1]):
        if val == nop:
            # Chooses states where all particles are in A
            psi_initial[idx] = e_vec[e_vec_num, j]
            j += 1

    return psi_initial / la.norm(psi_initial)


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


def density_matrix_b(label, e_vec, nos, nol_b, nop):
    """
    Calculates density matrix for sub-lattice B. Error checks density matrix
    whose trace should be 1.0. Warning raised if trace differs by >0.1.

    :param label: Relabelled states
    :param e_vec: Eigenvectors
    :param nos: No. of states
    :param nol_b: No. of lattice sites in B
    :param nop: No. of particles
    :return: Density matrix of sub-lattice B
    :return: Trace of square of density matrix

    """
    dim_b = sum_ncr(nol_b, nop + 1)
    rho_b = np.zeros(shape=(dim_b, dim_b), dtype=complex, order='F')

    for i in range(nos):
        for j in range(nos):
            if label[i, 1] == label[j, 1] and label[i, 0] == label[j, 0]:
                m = int(label[i, 2] + sum_ncr(nol_b, (nop - label[i, 1])) - 1)
                n = int(label[j, 2] + sum_ncr(nol_b, (nop - label[j, 1])) - 1)

                rho_b[m, n] += np.vdot(e_vec[j], e_vec[i])

    tr_rho = np.trace(rho_b, dtype=float)

    if mt.fabs(tr_rho - 1.0) > 1.0e-1:
        warnings.warn('Trace of density matrix B not 1, Trace=', tr_rho)

    return rho_b


def trace_squared(rho):
    """
    Calculate the trace of the square of the density matrix.
    :param rho: Density matrix
    :return: Trace of the square of the density matrix
    """
    return np.trace(np.linalg.matrix_power(rho, 2))


def vn_entropy_b(psi_t, label, nos, nol_b, nop):
    """
    Calculates Von-Neumann entropy as entropy = - tr(rho * ln(rho)).
    Filter used to suppress warning 'The logm input matrix may be nearly
    singular'.
    :param psi_t: Psi(t)
    :param label: Relabelled states
    :param nos: No. of states
    :param nol_b: No. of lattice sites in B
    :param nop: No. of particles
    :return: Real Von-Neumann entropy
    :return: Trace of density matrix of B

    """
    vn_entropy = np.zeros(len(psi_t), dtype=complex, order='F')
    tr_sqr = np.zeros(len(psi_t), dtype=float)

    warnings.filterwarnings('ignore')

    idx = 0
    for val in tqdm(psi_t):
        d_matrix = density_matrix_b(label, val, nos, nol_b, nop)
        vn_entropy[idx] = - np.trace(np.dot(d_matrix, la.logm(d_matrix)))
        tr_sqr[idx] = trace_squared(d_matrix)
        idx += 1

    return vn_entropy.real, tr_sqr


def time_evolution(psi_0, h, nos, timesteps):
    """
    Psi evolved as |Psi(t)> = exp(-i * H * t)|Psi(0)>
    :param psi_0: Initial state
    :param h: Hamiltonian matrix
    :param nos: No. of states
    :param timesteps: Array of times
    :return: Array of Psi(t)

    """
    psi_t = np.zeros(shape=(len(timesteps), nos), dtype=complex, order='F')

    idx = 0
    for t in tqdm(timesteps):
        psi_t[idx] = np.dot(la.expm(-1.0j * h * t), psi_0)
        idx += 1

    return psi_t


def avg_particles(psi_t, timesteps, labels, nop):
    """
    Counts the average number of particles in sub-lattices A and B.
    :param psi_t:
    :param timesteps: Array of times
    :param labels:
    :param nop: Total no. of particles
    :return: Avg. particles in sub-lattice A
    :return: Avg. particles in sub-lattice B

    """
    avg_a = np.zeros_like(timesteps, dtype=np.int32)
    avg_b = np.zeros_like(timesteps, dtype=np.int32)

    for idx in range(len(timesteps)):
        for idx2, val in enumerate(labels[:, 1]):
            fraction = (np.vdot(psi_t[idx, idx2], psi_t[idx, idx2])).real
            avg_a[idx] += fraction * val
            avg_b[idx] += fraction * (nop - val)

    return avg_a, avg_b
