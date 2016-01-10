from __future__ import division

import warnings

import numpy as np
import scipy.linalg as la

cimport cython
cimport numpy as np

import Main
import Output
import SubRoutine2


# Returns a random eigenvector by placing eigenvectors from A in a zero matrix
@cython.boundscheck(False)
def random_eigenvector(np.ndarray e_vecs, np.ndarray relabelled_states, int nos, int nos_a, int nop):
    psi_initial = np.zeros(nos, dtype=np.complex)
    rand = np.random.randint(0, nos_a)
    j = 0
    for i in relabelled_states[1]:
        if i == nop:
            psi_initial[i] = e_vecs[rand][j]
            j += 1

    # print "\nEigenvector", rand, "chosen randomly for time evolution"
    a = la.norm(psi_initial)
    psi_initial /= a
    return psi_initial


# Calculates Von-Neumann entropy as entropy = - tr(rho * ln(rho))
@cython.boundscheck(False)
def von_neumann_b(np.ndarray psi_array, np.ndarray relabelled_states, int nos):
    entropy_b = np.zeros(len(psi_array), dtype=np.complex)
  
    for idx, psi_val in enumerate(psi_array):
        d_matrix_b = SubRoutine2.density_matrix_b(relabelled_states, psi_val, nos)
        entropy_b[idx] = -1.0 * np.trace(np.dot(d_matrix_b, la.logm(d_matrix_b)))
        # Output.write_file('d_matrix_b_{}.csv'.format(idx), d_matrix_b)

    return entropy_b

@cython.boundscheck(False)
def von_neumann_a(np.ndarray psi_array, np.ndarray relabelled_states, int nos):
    entropy_a = np.zeros(len(psi_array), dtype=np.complex)
 
    for idx, psi_val in enumerate(psi_array):
        d_matrix_a = SubRoutine2.density_matrix_a(relabelled_states, psi_val, nos)
        entropy_a[idx] = -1.0 * np.trace(np.dot(d_matrix_a, la.logm(d_matrix_a)))
        # Output.write_file('d_matrix_b_{}.csv'.format(idx), d_matrix_a)

    return entropy_a


# Psi evolved as |psi(t)> = exp(-i * H * t)|psi(0)>
@cython.boundscheck(False)
def time_evolution(np.ndarray psi_initial, np.ndarray hamiltonian, int nos):
    s = Main.System()

    timestep_array = np.arange(s.t_initial, s.t_final, s.delta_t)
    psi_t = np.zeros(shape=(len(timestep_array), nos), dtype=np.complex)
    
    for idx, t in enumerate(timestep_array):
        psi_t[idx] = np.dot(la.expm(1.0j * hamiltonian * t), psi_initial)
        
        norm = la.norm(psi_t[idx]) 
        if norm - 1.0 > 1.0e-5:
            Output.warning('Norm of Psi not 1, norm=', norm)
        # print('norm = ', la.norm(psi_t[idx]))

    return psi_t, timestep_array
