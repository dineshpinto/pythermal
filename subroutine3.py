# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Time evolving hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics
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


def sum_ncr(n, k):
    """
    Calculates nC0 + nC1 + ... + nCr
    :param n: Total no. of items
    :param k: No. of items chosen
    :return: Sum of combinations

    """
    s = 0
    f = mt.factorial

    # Find nCr using floor/integer division
    for r in range(k):
        s += f(n) // (f(r) * f(n - r))

    return s


# Calculates the density matrix for sub-lattice A
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
    # den_trace_a2 = np.trace(np.linalg.matrix_power(density_mat_a, 2))

    # Error checking to make sure trace of DM remains ~1.0
    if mt.fabs(den_trace_a - 1.0) > 1.0e-5:
        output.warning('Trace of density matrix A is not 1, Trace=', den_trace_a)

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
    den_trace_b2 = np.trace(np.linalg.matrix_power(density_mat_b, 2))

    # Error checking to make sure trace of DM remains ~1.0
    if mt.fabs(den_trace_b - 1.0) > 1.0e-1:
        output.warning('Trace of density matrix B is not 1, Trace=', den_trace_b)

    return density_mat_b, den_trace_b2


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
    trace2_b = np.zeros(len(psi_array), dtype=np.float)

    # [CAUTION] Replaces default warning(Below) and ignores future warnings within scope
    # Output.warning('The logm input matrix may be nearly singular')
    warnings.filterwarnings('ignore')

    idx = 0
    for psi_val in tqdm.tqdm(psi_array):
        d_matrix_b, trace2_b[idx] = density_matrix_b(relabelled_states, psi_val, nos, nol_b, nop)
        entropy_b[idx] = -1.0 * np.trace(np.dot(d_matrix_b, la.logm(d_matrix_b)))
        idx += 1

    return entropy_b.real, trace2_b


def time_evolution(psi_initial, hamiltonian, nos, timestep_array, relabel_states, nop):
    """
    Psi evolved as |Psi(t)> = exp(-i * H * t)|Psi(0)>
    :param psi_initial: Initial state
    :param hamiltonian: Hamiltonian matrix
    :param nos: No. of states
    :param timestep_array: Arrays of times
    :param relabel_states: Relabelled states
    :param nop: No. of particles
    :return: Array of Psi(t)
    :return: Avg. particles in sub-lattice A
    :return: Avg. particles in sub-lattice B

    """
    psi_t = np.zeros(shape=(len(timestep_array), nos), dtype=np.complex, order='F')
    sum_a = np.zeros_like(timestep_array)
    sum_b = np.zeros_like(timestep_array)

    idx = 0
    for t in tqdm.tqdm(timestep_array):

        psi_t[idx] = np.dot(la.expm(-1.0j * hamiltonian * t), psi_initial)

        for idx2, val in enumerate(relabel_states[:, 1]):
            sum_a[idx] += (np.vdot(psi_t[idx, idx2], psi_t[idx, idx2])).real * val
            sum_b[idx] += (np.vdot(psi_t[idx, idx2], psi_t[idx, idx2])).real * (nop - val)

        idx += 1

    return psi_t, sum_a, sum_b
