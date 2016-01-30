# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Time evolving hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group
# St. Stephen's Centre for Theoretical Physics
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import division, print_function

import math as mt
import warnings

import numpy as np
import scipy.linalg as la
import tqdm

try:
    from builtins import range
except ImportError:
    from __builtin__ import range

import output


def ncr(n, r):
    """
    Find nCr using floor/integer division
    :param n: Total no. of items
    :param r: No. of items chosen
    :return: No. of combinations

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
    s = sum(ncr(n, r) for r in range(k))
    return s


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
        output.warning('Trace of density matrix A is not 1, Trace=',
                       den_trace_a)

    return density_mat_a


def density_matrix_b(label, e_vec, nos, nol_b, nop):
    """
    Calculates density matrix for sub-lattice B
    :param label: Relabelled states
    :param e_vec: Eigenvectors
    :param nos: No. of states
    :param nol_b: No. of lattice sites in B
    :param nop: No. of particles
    :return: Density matrix of sub-lattice B
    :return: Trace of square of density matrix

    """
    dim_b = sum_ncr(nol_b, nop + 1)
    density_mat_b = np.zeros(shape=(dim_b, dim_b), dtype=np.complex, order='F')

    for i in range(nos):

        for j in range(nos):

            if label[i, 1] == label[j, 1] and label[i, 0] == label[j, 0]:
                m = int(label[i, 2] + sum_ncr(nol_b, (nop - label[i, 1])) - 1)
                n = int(label[j, 2] + sum_ncr(nol_b, (nop - label[j, 1])) - 1)

                density_mat_b[m, n] += np.vdot(e_vec[j], e_vec[i])

    # Calculate trace & trace square of density matrix B
    den_trace_b = np.trace(density_mat_b.real)

    # Error checking to make sure trace of DM remains ~1.0
    if mt.fabs(den_trace_b - 1.0) > 1.0e-1:
        output.warning('Trace of density matrix B is not 1, Trace=',
                       den_trace_b)

    return density_mat_b


def trace_squared(density_matrix):
    """
    Calculate the trace of the square of the density matrix.
    :param density_matrix: Density matrix
    :return: Trace of the square of the density matrix

    """
    den_trace_squared = np.trace(np.linalg.matrix_power(density_matrix, 2))
    return den_trace_squared


def von_neumann_b(psi_array, relabelled_states, nos, nol_b, nop):
    """
    Calculates Von-Neumann entropy as entropy = - tr(rho * ln(rho))
    :param psi_array: Psi(t)
    :param relabelled_states: Relabelled states
    :param nos: No. of states
    :param nol_b: No. of lattice sites in B
    :param nop: No. of particles
    :return: Real Von-Neumann entropy
    :return: Trace of density matrix of B

    """
    entropy_b = np.zeros(len(psi_array), dtype=np.complex, order='F')
    trace_squared_b = np.zeros(len(psi_array), dtype=np.float)

    # [CAUTION] Replaces default warning(Below) and ignores future warnings
    # within von_neumann_b
    # Output.warning('The logm input matrix may be nearly singular')
    warnings.filterwarnings('ignore')

    idx = 0
    for psi_val in tqdm.tqdm(psi_array):
        d_matrix_b = density_matrix_b(relabelled_states, psi_val, nos, nol_b,
                                      nop)
        entropy_b[idx] = -1.0 * np.trace(np.dot(d_matrix_b,
                                                la.logm(d_matrix_b)))
        trace_squared_b[idx] = trace_squared(d_matrix_b)
        idx += 1

    return entropy_b.real, trace_squared_b


def time_evolution(psi_initial, hamiltonian, nos, timestep_array,
                   relabel_states, nop):
    """
    Psi evolved as |Psi(t)> = exp(-i * H * t)|Psi(0)>
    :param psi_initial: Initial state
    :param hamiltonian: Hamiltonian matrix
    :param nos: No. of states
    :param timestep_array: Array of times
    :param relabel_states: Relabelled states
    :param nop: No. of particles
    :return: Array of Psi(t)
    :return: Avg. particles in sub-lattice A
    :return: Avg. particles in sub-lattice B

    """
    psi_t = np.zeros(shape=(len(timestep_array), nos), dtype=np.complex,
                     order='F')
    sum_a = np.zeros_like(timestep_array)
    sum_b = np.zeros_like(timestep_array)

    idx = 0
    for t in tqdm.tqdm(timestep_array):

        psi_t[idx] = np.dot(la.expm(-1.0j * hamiltonian * t), psi_initial)

        for idx2, val in enumerate(relabel_states[:, 1]):
            sum_a[idx] += \
                (np.vdot(psi_t[idx, idx2], psi_t[idx, idx2])).real * val
            sum_b[idx] += \
                (np.vdot(psi_t[idx, idx2], psi_t[idx, idx2])).real * (nop -
                                                                      val)

        idx += 1

    return psi_t, sum_a, sum_b
