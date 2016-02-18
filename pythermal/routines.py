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
import traceback
import warnings

import numpy as np
import scipy.linalg as la
from tqdm import tqdm

try:
    from builtins import range
except ImportError:
    from __builtin__ import range

__all__ = ['position_states', '_hamiltonian', 'distribute',
           'hamiltonian_parallel', 'eig', 'ncr', 'sum_ncr', 'relabel',
           'initial_state', 'density_matrix_a', 'density_matrix_b',
           'trace_squared', 'vn_entropy', 'time_evolution',
           'avg_particles', 'state_initializer', 'epsilon']


def position_states(lat, nop, del_pos=None):
    """
    Returns eigenstates for a given lattice & particles after deleting sites
    :param lat: Array of lattice sites
    :param nop: Nop of particles in lattice
    :param del_pos: Lattice sites to delete
    :return: Eigenstates
    :return: Total no. of states

    """
    if del_pos is None:
        pos_states = np.array(list(it.combinations(lat, nop)), dtype=np.int32)
    else:
        lat_del = np.delete(lat, del_pos - 1)
        pos_states = np.array(list(it.combinations(lat_del, nop)),
                              dtype=np.int32)

    return pos_states, len(pos_states)


def _hamiltonian(start, stop, nos, ndims, nop, e_states, queue, h):
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


def hamiltonian_parallel(lattice, ndims, nop):
    """
    Multiple parallel calls to hamiltonian_2d.
    :param lattice: Lattice
    :param ndims: Shape of lattice.
    :param nop: No. of particles
    :return: Hamiltonian matrix

    """
    pos_states, nos = position_states(lattice, nop)
    process_list = []
    queue = mp.Queue()  # Setting up a queue to store each processes' output
    h = np.zeros(shape=(nos, nos), dtype=np.float32)

    # No. of processes to create for parallel processing
    n_processes = mp.cpu_count()

    for i in range(n_processes):
        start, stop = distribute(nos, n_processes, i)
        args = (start, stop, nos, ndims, nop, pos_states, queue, h)
        process = mp.Process(target=_hamiltonian, args=args)
        process_list.append(process)  # Create list of processes
        process.start()

    for i in range(n_processes):  # Retrieves output from queue
        h += queue.get()

    queue.close()
    queue.join_thread()

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
        print('Trace of density matrix B not 1, Trace=', tr_rho)

    return rho_b


def trace_squared(rho):
    """
    Calculate the trace of the square of the density matrix.
    :param rho: Density matrix
    :return: Trace of the square of the density matrix
    """
    return np.trace(np.linalg.matrix_power(rho, 2))


def vn_entropy(psi_t, label, nos, nol_b, nop):
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
    entropy_b = np.zeros(len(psi_t), dtype=complex, order='F')
    tr_sqr = np.zeros(len(psi_t), dtype=float)

    warnings.filterwarnings('ignore')

    idx = 0
    for val in tqdm(psi_t):
        d_matrix = density_matrix_b(label, val, nos, nol_b, nop)
        entropy_b[idx] = - np.trace(np.dot(d_matrix, la.logm(d_matrix)))
        tr_sqr[idx] = trace_squared(d_matrix)
        idx += 1

    return entropy_b.real, tr_sqr


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
    avg_a = np.zeros_like(timesteps)
    avg_b = np.zeros_like(timesteps)

    for idx in range(len(timesteps)):
        for idx2, val in enumerate(labels[:, 1]):
            fraction = (np.vdot(psi_t[idx, idx2], psi_t[idx, idx2])).real
            avg_a[idx] += fraction * val
            avg_b[idx] += fraction * (nop - val)

    return avg_a, avg_b


def state_initializer(e_vals, e_vecs, num=None):
    """
    Returns a set of normalized eigenvectors. If num is not specified uses
    average difference between eigenvalues to evenly space out eigenvalues.
    If num is specified, just generates a list of size [num] of eigenvalues.
    :param e_vals: Eigenvalues
    :param e_vecs: Eigenvectors
    :param num: Number of eigenvalues (optional)
    :return: Normalized initial states

    """
    if num is not None:
        length = len(e_vals)

        if num < 2:
            raise ValueError('No. of eigenstates chosen is too small[{}]. '
                             'Min is 2'.format(num))

        if num > length:
            raise ValueError('Too many eigenstates chosen[{}]. Max is [{}].'
                             .format(num, length))
        # step = length // num
        # idx_array = [i for i in range(0, len(e_vals) - step, step)]
        idx_array = (np.random.triangular(0, length // 2, length, num)).\
            astype(int)
        idx_array = sorted(idx_array)
    else:
        eps = epsilon(e_vals)

        idx_array = [0]
        a = e_vals[0]
        for idx, eigenvalue in enumerate(e_vals):
            if eigenvalue - a >= eps:
                idx_array.append(idx)
                a = eigenvalue

    initial_states = [e_vecs[:, idx] / la.norm(e_vecs[:, idx]) for idx in
                      idx_array]
    initial_eigenvalues = [e_vals[idx] for idx in idx_array]
    return np.array(initial_states), (initial_eigenvalues, idx_array)


def epsilon_alt(eigenvalues):
    """
    Defines an epsilon based on the average distance between
    eigenvalues.
    :param eigenvalues: Eigenvalues of system

    """
    diff = [0] * (len(eigenvalues) - 1)

    for idx in range(len(eigenvalues) - 1):
        diff[idx] = abs(eigenvalues[idx + 1] - eigenvalues[idx])

    return sum(diff) / len(diff)


def epsilon(eigenvalues):
    """
    Defines an epsilon based on the average distance between
    eigenvalues. Approx twice as fast as epsilon().
    :param eigenvalues: Eigenvalues
    :return: Average difference between sucessive eigenvalues

    """
    iterable = iter(eigenvalues)
    prev = next(iterable)

    s = 0
    for element in iterable:
        s += element - prev
        prev = element

    return s / len(eigenvalues)


def h_block_diagonal(lat_b, n_dim):
    hb_0 = hamiltonian_parallel(lat_b, n_dim, 0)
    yield hb_0
    hb_1 = hamiltonian_parallel(lat_b, n_dim, 1)
    yield hb_1
    hb_2 = hamiltonian_parallel(lat_b, n_dim, 2)
    yield hb_2
    # hb_3 = hamiltonian_parallel(lat_b, n_dim, 3)
    # yield hb_3
    # hb_4 = hamiltonian_parallel(lat_b, n_dim, 4)
    # yield hb_4
    # hb_5 = hamiltonian_parallel(lat_b, n_dim, 5)
    # yield hb_5
    # hb_6 = hamiltonian_parallel(lat_b, n_dim, 6)
    # yield hb_6

#
# def transform(rho_pbasis, hbd_evecs):
#     print (rho_pbasis.shape, hbd_evecs.shape)
#     exit(0)
#     print (np.vdot(hbd_evecs, np.dot(rho_pbasis, hbd_evecs))
# )
#     return np.vdot(hbd_evecs, np.dot(rho_pbasis, hbd_evecs))



def transformation(pho_pbasis, e_vecs_bd):
    """

    :param pho_pbasis:
    :param e_vecs_bd:
    :return:
    """
    rho_ebasis = np.zeros_like(pho_pbasis, dtype=complex)

    for i in range(pho_pbasis.shape[0]):
        for j in range(pho_pbasis.shape[1]):
            rho_ebasis[i, j] = np.vdot(e_vecs_bd[:, i], np.dot(pho_pbasis, e_vecs_bd[:, j]))

    if not np.allclose(np.transpose(np.conjugate(rho_ebasis)), rho_ebasis):
        print("Transformation matrix is not symmetric")

    return rho_ebasis
                                                                                                                                                                                                                                                                                                                                                                                                                                                            