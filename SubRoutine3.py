from __future__ import division

import warnings

import numpy as np
import scipy.linalg as la
import tqdm

import Main
import Output
import SubRoutine2


# Returns a random eigenvector by placing eigenvectors from A in a zero matrix
def random_eigenvector(e_vecs, relabelled_states, nos, nos_a, nop):
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
def von_neumann_b(psi_array, relabelled_states, nos):
    entropy_b = np.zeros(len(psi_array), dtype=np.complex)

    # Replaces default warning
    Output.warning('The logm input matrix may be nearly singular')
    warnings.filterwarnings('ignore')

    for idx, psi_val in tqdm.tqdm(enumerate(psi_array)):
        d_matrix_b = SubRoutine2.denmatrix_b(relabelled_states, psi_val, nos)
        entropy_b[idx] = -1.0 * np.trace(np.dot(d_matrix_b, la.logm(d_matrix_b)))

    print
    return entropy_b


# Psi evolved as |psi(t)> = exp(-i * H * t)|psi(0)>
def time_evolution(psi_initial, hamiltonian, nos):
    s = Main.System()

    timestep_array = np.arange(s.t_initial, s.t_final, s.delta_t)
    psi_t = np.zeros(shape=(len(timestep_array), nos), dtype=np.complex)

    for idx, t in tqdm.tqdm(enumerate(timestep_array)):
        psi_t[idx] = np.dot(la.expm(-1.0j * hamiltonian * t), psi_initial)

    return psi_t, timestep_array
