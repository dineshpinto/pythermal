# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Thermal equilibrium of hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group
# St. Stephen's Centre for Theoretical Physics, St. Stephen's College, Delhi
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
import scipy.linalg as la
from tqdm import tqdm

try:
    # Python 3
    from builtins import range
except ImportError:
    # Python 2
    from __builtin__ import range

__all__ = ['position_states', '_hamiltonian_', 'distribute',
           'hamiltonian_parallel', 'diagonalize', 'ncr', 'sum_ncr',
           'relabel', 'rho_a_pbasis', 'rho_b_pbasis',
           'h_block_diagonal', 'transform_basis', 'naive_thermal',
           'initial_sublattice_state', 'trace_squared', 'vn_entropy_b',
           'time_evolution', 'avg_particles']


def position_states(lat, nop, del_pos=None):
    """
    Returns position states for a given lattice and number of particles.
    Parameter del_pos can be used to delete lattice sites.

    :param lat: Array of lattice sites
    :param nop: Nop of particles in lattice
    :param del_pos: Lattice sites to delete
    :return: Positions states
    :return: Total no. of states
    """
    if del_pos is None:
        pos_states = np.array(list(it.combinations(lat, nop)), dtype=np.int32)
    else:
        del_pos = np.array(del_pos)
        lat_del = np.delete(lat, del_pos - 1)
        pos_states = np.array(list(it.combinations(lat_del, nop)),
                              dtype=np.int32)

    return pos_states, len(pos_states)


def _hamiltonian_(start, stop, nos, ndims, nop, pos_states):
    """
    Core Hamiltonian function.

    Generates hamiltonian matrix for a given system. Uses global
    multiprocessing array (shared memory array) to store output hamiltonian.

    When coupled to its function wrapper (hamiltonian_parallel), n processes
    (n is no. of CPU's) call _hamiltonian_ simultaneously. The distribute
    function allocated tasks to each process. A task is defined by the
    starting and stopping points of the outer loop (represented by 'j').

    :param start: Starting point of [j] iteration
    :param stop: Stopping point of [j] iteration
    :param nos: No. of states
    :param ndims: No. of dimension
    :param nop: No. of particles
    :param pos_states: Position states
    """
    for j in tqdm(range(start, stop)):
        for k in range(nos):
            # Find common elements and their size
            c = np.intersect1d(pos_states[j], pos_states[k])
            c_size = np.size(c)

            if c_size == nop - 1:
                c_sum = np.sum(c, dtype=np.int32)
                j_sum = np.sum(pos_states[j], dtype=np.int32)
                k_sum = np.sum(pos_states[k], dtype=np.int32)

                # Differ by one element
                if abs(j_sum - k_sum) == ndims:
                    # Differ by dimension of lattice
                    ham[j, k] = 1
                elif (k_sum - j_sum) == 1 and not (j_sum - c_sum) % ndims == 0:
                    # Sites next to each other and not at edge
                    ham[j, k] = 1
                elif (j_sum - k_sum) == 1 and not (j_sum - c_sum) % ndims == 1:
                    # Sites next to each other and not at edge
                    ham[j, k] = 1


def distribute(n_items, n_processes, i):
    """
    Defines a starting and stopping point for a particular task to be
    allocated to a process. Returns a (start, stop) tuple.

    :param n_items: Total no. of items
    :param n_processes: Total no. of processes
    :param i: Process no. (not same as PID)
    :return: Start & Stop point index in no. of items
    """
    items_per_process = n_items // n_processes
    start = i * items_per_process

    if i == n_processes - 1:
        # For last process, appends all remaining tasks to last core
        stop = n_items
    else:
        stop = items_per_process * (i + 1)

    return start, stop


def hamiltonian_parallel(lattice, ndims, nop):
    """
    Wrapper for _hamiltonian. Creates multiple processes, each of which
    calls _hamiltonian simultaneously. Defines a global 'ham', which is a
    multiprocessing array interfaced with ctypes and reshaped to generate an
    empty (filled with zeros) hamiltonian array.

    :param lattice: Lattice used
    :param ndims: Dimensionality of lattice
    :param nop: No. of particles
    :return: Hamiltonian matrix
    """
    pos_states, nos = position_states(lattice, nop)

    # Creates a multiprocessing Array and reshapes it for the hamiltonian
    # The Array is declared globally within this function
    # The Array is a part of shared memory for various processes
    global ham
    # Create wrapper for ctype array
    ham_base = mp.Array(ctypes.c_int8, nos * nos)
    # Extract object from wrapper and cast as numpy.ndarray
    ham = np.ctypeslib.as_array(ham_base.get_obj())
    ham = ham.reshape(nos, nos)

    # No. of processes to create for parallel processing
    n_processes = mp.cpu_count()
    process_list = []

    # Define and create processes
    for i in range(n_processes):
        start, stop = distribute(nos, n_processes, i)
        args = (start, stop, nos, ndims, nop, pos_states)
        process = mp.Process(target=_hamiltonian_, args=args)
        process_list.append(process)
        process.start()

    # Join processes together
    for processes in process_list:
        processes.join()

    return ham


def diagonalize(h):
    """
    Calculates eigenvectors and eigenvalues used Pade algorithm (see SciPy
    documentation).

    Control threads using OpenMP. Link to lower level OpenBLAS (Basic Linear
    Algebra Subroutines) written in Fortran for parallel processing.

    :param h: Matrix
    :return: Real array of eigenvalues
    :return: Complex array of eigenvectors
    """
    eigenvalues, eigenvectors = la.eig(h, check_finite=False)

    # Select only real part of eigenvalues
    eigenvalues = eigenvalues.real

    # Sort eigenvalues and eigenvectors by ascending eigenvalue
    index = eigenvalues.argsort()
    eigenvectors = eigenvectors[:, index]
    eigenvalues = eigenvalues[index]

    return eigenvalues, eigenvectors


def ncr(n, r):
    """
    No. of combinations of k items taken from n items.

    :param n: Total no. of items
    :param r: No. of items chosen
    :return: Total no. of combinations
    """
    f = mt.factorial
    return f(n) // (f(r) * f(n - r))


def sum_ncr(n, k):
    """
    Calculates nC0 + nC1 + ... + nCr.

    :param n: Total no. of items
    :param k: No. of items chosen
    :return: Sum of combinations
    """
    return sum(ncr(n, r) for r in range(k))


def relabel(e_states, nop, nol_b, lat_a):
    """
    The state is |\psi>AB = \Sigma_{i, n, \mu} a_i_n \mu |i_n\mu>
    where the product state is written as |i_n\mu>.
    Refer to "The Density Matrix using Relabelled States.pdf"
     
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


def rho_a_pbasis(label, e_vec, nos, nol_a, nop):
    """
    Calculates density matrix for sub-lattice A. Error checks density matrix
    (trace should be 1.0). Warning raised if trace differs by 1(+-)0.1.


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
    (trace should be 1.0). Warning raised if trace differs by 1(+-)0.1.

    :param label: Relabelled states
    :param e_vec: Eigenvectors
    :param nos: No. of states
    :param nol_b: No. of lattice sites in B
    :param nop: No. of particles
    :return: Density matrix of sub-lattice B
    """
    dim_b = sum_ncr(nol_b, nop + 1)
    rho_b = np.zeros(shape=(dim_b, dim_b), dtype=complex)

    for i in tqdm(range(nos)):
        for j in range(nos):
            if label[i, 1] == label[j, 1] and label[i, 0] == label[j, 0]:
                m = int(label[i, 2] + sum_ncr(nol_b, (nop - label[i, 1])) - 1)
                n = int(label[j, 2] + sum_ncr(nol_b, (nop - label[j, 1])) - 1)
                rho_b[m, n] += np.vdot(e_vec[j], e_vec[i])

    tr_rho = np.trace(rho_b.real, dtype=float)

    if mt.fabs(tr_rho - 1.0) > 1.0e-1:
        print('WARNING: Trace of density matrix B not 1, Trace=', tr_rho)

    return rho_b


def h_block_diagonal(lat_b, n_dim, nop):
    """
    Creates a block diagonal matrix containing the hamiltonian (for various
    no. of particles) of B placed in blocks along the diagonal.

    :param lat_b: Lattice sites in B
    :param n_dim: No. of dimensions
    :param nop: No. of particles
    :return: Block diagonal matrix
    """
    bd = []
    for i in range(nop + 1):
        bd.append(hamiltonian_parallel(lat_b, n_dim, i))
    return la.block_diag(*bd)


def transform_basis(rho_pbasis, e_vecs_bd):
    """
    Transforms DM from position basis to DM in energy basis.

    :param rho_pbasis: Rho in position basis
    :param e_vecs_bd: Eigenvectors of block diagonal hamiltonian
    :return: Rho in energy basis
    """
    rho_ebasis = np.zeros_like(rho_pbasis, dtype=complex)

    for i in tqdm(range(rho_pbasis.shape[0])):
        for j in range(rho_pbasis.shape[1]):
            rho_ebasis[i, j] = \
                np.vdot(e_vecs_bd[:, i], np.dot(rho_pbasis, e_vecs_bd[:, j]))

    return rho_ebasis


def naive_thermal(rho):
    """
    Compare maximum diagonal and off diagonal terms in density matrix.
    Note: Makes copy of DM (bypass will destroy original DM)

    :param rho: Density matrix in energy basis
    :return: Max diagonal element
    :return: Max off-diagonal element
    """
    max_diag = rho.diagonal().max()

    rho_copy = np.copy(rho)
    np.fill_diagonal(rho_copy, -np.inf)
    max_offdiag = rho_copy.max()

    return max_diag, max_offdiag


def initial_sublattice_state(e_vec, label, nos, nop, e_vec_num):
    """
    Returns a normalized initial state with all particles in one sub-lattice.
    Control which sub-lattice is used by passing appropriate eigenvectors
    (i.e. eigenvectors of the chosen sub-lattice) as an argument.

    :param e_vec: Eigenvectors of either sub-lattice
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
            # Picks out states where all particles are in one sub-lattice
            psi_initial[idx] = e_vec[e_vec_num, j]
            j += 1

    return psi_initial / la.norm(psi_initial)


def trace_squared(rho):
    """
    Calculate the trace of the square of the density matrix.

    :param rho: Density matrix
    :return: Trace of the square of the density matrix
    """
    return np.trace(np.linalg.matrix_power(rho, 2))


def vn_entropy_a(psi_t, label, nos, nol_a, nop):
    """
    Calculates Von-Neumann entropy as S = - tr(rho * ln(rho)).
    Also calculates trace of square of density matrix (measure of
    entanglement).
    Uses a filter to suppress 'WARNING: The logm input matrix may be nearly
    singular'. 

    :param psi_t: Psi(t)
    :param label: Relabelled states
    :param nos: No. of states
    :param nol_a: No. of lattice sites in A
    :param nop: No. of particles
    :return: Real Von-Neumann entropy
    :return: Trace of density matrix of B
    """
    vn_entropy = np.zeros(len(psi_t), dtype=complex)
    tr_sqr = np.zeros(len(psi_t), dtype=float)

    warnings.filterwarnings('ignore')

    idx = 0
    for val in tqdm(psi_t):
        d_matrix = rho_a_pbasis(label, val, nos, nol_a, nop)
        vn_entropy[idx] = - np.trace(np.dot(d_matrix, la.logm(d_matrix)))
        tr_sqr[idx] = trace_squared(d_matrix)
        idx += 1

    return vn_entropy.real, tr_sqr


def vn_entropy_b(psi_t, label, nos, nol_b, nop):
    """
    Calculates Von-Neumann entropy as S = - tr(rho * ln(rho)).
    Also calculates trace of square of density matrix (measure of
    entanglement).
    Uses a filter to suppress 'WARNING: The logm input matrix may be nearly
    singular'. Wraps loop in tqdm for progress bar.

    :param psi_t: Psi(t)
    :param label: Relabelled states
    :param nos: No. of states
    :param nol_b: No. of lattice sites in B
    :param nop: No. of particles
    :return: Real Von-Neumann entropy
    :return: Trace of density matrix of B
    """
    vn_entropy = np.zeros(len(psi_t), dtype=complex)
    tr_sqr = np.zeros(len(psi_t), dtype=float)

    warnings.filterwarnings('ignore')

    for idx, vec in enumerate(psi_t):
        d_matrix = rho_b_pbasis(label, vec, nos, nol_b, nop)
        vn_entropy[idx] = - np.trace(np.dot(d_matrix, la.logm(d_matrix)))
        tr_sqr[idx] = trace_squared(d_matrix)

    return vn_entropy.real, tr_sqr


def time_evolution(psi_0, h, nos, timesteps):
    """
    Psi evolved as |Psi(t)> = exp(-i * h * t)|Psi(0)>.
    Uses matrix exponential for computation.

    :param psi_0: Initial state
    :param h: Hamiltonian matrix
    :param nos: No. of states
    :param timesteps: Array of times
    :return: Array of Psi(t)
    """
    psi_t = np.zeros(shape=(len(timesteps), nos), dtype=complex)

    idx = 0
    for t in tqdm(timesteps):
        psi_t[idx] = np.dot(la.expm(-1.0j * h * t), psi_0)
        idx += 1

    return psi_t


def avg_particles(psi_t, timesteps, labels, nop):
    """
    Calculates the average number of particles in sub-lattices A and B.

    :param psi_t: Array of psi at various times
    :param timesteps: Array of times
    :param labels: Relabelled states
    :param nop: Total no. of particles
    :return: Avg. particles in sub-lattice A
    :return: Avg. particles in sub-lattice B
    """
    avg_a, avg_b = np.zeros_like(timesteps), np.zeros_like(timesteps)

    for idx in range(len(timesteps)):
        for idx2, val in enumerate(labels[:, 1]):
            fraction = (np.vdot(psi_t[idx, idx2], psi_t[idx, idx2])).real
            avg_a[idx] += fraction * val
            avg_b[idx] += fraction * (nop - val)

    return avg_a, avg_b
