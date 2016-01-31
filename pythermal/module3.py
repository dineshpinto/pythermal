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

__all__ = ['ncr', 'sum_ncr', 'density_matrix_a', 'density_matrix_b',
           'trace_squared', 'von_neumann_b', 'time_evolution',
           'particle_counter']


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
    density_mat_b = np.zeros(shape=(dim_b, dim_b), dtype=complex, order='F')

    for i in range(nos):

        for j in range(nos):

            if label[i, 1] == label[j, 1] and label[i, 0] == label[j, 0]:
                m = int(label[i, 2] + sum_ncr(nol_b, (nop - label[i, 1])) - 1)
                n = int(label[j, 2] + sum_ncr(nol_b, (nop - label[j, 1])) - 1)

                density_mat_b[m, n] += np.vdot(e_vec[j], e_vec[i])

    # Calculate trace & trace square of density matrix B
    den_trace_b = np.trace(density_mat_b, dtype=float)

    # Error checking to make sure trace of DM remains ~1.0
    if mt.fabs(den_trace_b - 1.0) > 1.0e-1:
        output.warning('Trace of density matrix B is not 1, Trace=',
                       den_trace_b)

    return density_mat_b


def trace_squared(a):
    """
    Calculate the trace of the square of the density matrix.
    :param a: Density matrix
    :return: Trace of the square of the density matrix

    """
    tr_squared = np.trace(np.linalg.matrix_power(a, 2))
    return tr_squared


def von_neumann_b(psi_t, label, nos, nol_b, nop):
    """
    Calculates Von-Neumann entropy as entropy = - tr(rho * ln(rho))
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

    # [CAUTION] Replaces default warning(Below) and ignores future warnings
    # within von_neumann_b
    # Output.warning('The logm input matrix may be nearly singular')
    warnings.filterwarnings('ignore')

    idx = 0
    for val in tqdm.tqdm(psi_t):
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
    for t in tqdm.tqdm(timesteps):
        psi_t[idx] = np.dot(la.expm(-1.0j * h * t), psi_0)
        idx += 1

    return psi_t


def particle_counter(psi_t, timesteps, labels, nop):
    """
    Counts the average number of particles in sub-lattices A and B.
    :param psi_t:
    :param timesteps: Array of times
    :param labels:
    :param nop: Total no. of particles
    :return: Avg. particles in sub-lattice A
    :return: Avg. particles in sub-lattice B

    """
    avg_particles_a = np.zeros_like(timesteps)
    avg_particles_b = np.zeros_like(timesteps)

    for idx in range(len(timesteps)):
        for idx2, val in enumerate(labels[:, 1]):

            fraction = (np.vdot(psi_t[idx, idx2], psi_t[idx, idx2])).real

            avg_particles_a[idx] += fraction * val
            avg_particles_b[idx] += fraction * (nop - val)

    return avg_particles_a, avg_particles_b
