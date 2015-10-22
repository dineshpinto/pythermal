import numpy as np
import scipy.linalg as la
# import tqdm

import SubRoutine2
import Main


# Returns a random eigenvector
def random_eigenvector(e_vec, lab, nos, nos_a, nop):
    psi_initial = np.zeros(nos, dtype=np.complex)
    rand = np.random.randint(0, nos_a)
    j = 0
    for i in lab[1]:
        if i == nop:
            psi_initial[i] = e_vec[rand][j]
            j += 1

    print "\nEigenvector", rand, "chosen randomly"
    a = la.norm(psi_initial)
    psi_initial /= a

    return psi_initial


# Calculates Von-Neumann entropy as entropy = - rho * ln(rho)
def von_neumann_b(psi_array, labels, nos):
    num_states = len(psi_array)
    entropy_b = np.zeros(num_states, dtype=np.complex)

    for idx, psi_val in enumerate(psi_array):
        d_matrix_b = SubRoutine2.denmatrix_b(labels, psi_val, nos)
        entropy_b[idx] = -1.0 * np.trace(np.dot(d_matrix_b, la.logm(d_matrix_b)))

    return entropy_b


# Psi evolved in accordance with 1D paper psi(t) = SIGMA_(i=0)^(n) [|E_i><E_i|psi(0)>exp(-i*E_i*t / hbar)]
def psi_t(e_vecs, e_vals, nos, psi_initial, t):
    psi = np.zeros(nos, dtype=np.complex)
    for i in xrange(nos):
        psi += np.exp(-1.0j * e_vals[i] * t) * e_vecs[:, i] * np.vdot(e_vecs[:, i], psi_initial)

    return psi


def time_evolution(eigenvectors, eigenvalues, psi_initial, nos):
    s = Main.System()

    delta_t = (s.t_final - s.t_initial) / s.t_steps
    timestep_array = np.arange(s.t_initial, s.t_final, delta_t)
    psi_array = np.zeros(shape=(len(timestep_array), nos), dtype=np.complex)

    for idx, t in enumerate(timestep_array):
        psi_array[idx] = psi_t(eigenvectors, eigenvalues, nos, psi_initial, t)

    return psi_array, timestep_array


'''
# Psi evolved as psi(t) = psi(0) * exp(i * H * t)
def psi_t(psi_initial, hamiltonian, timestep):
    psi = np.dot(la.expm((-1.0e-34 * 1.0j * timestep * hamiltonian)), psi_initial)

    return psi


def time_evolution(psi_initial, hamiltonian, nos):
    s = Main.System()

    delta_t = (s.t_final - s.t_initial) / s.t_steps
    timestep_array = np.arange(s.t_initial, s.t_final, delta_t)
    psi_array = np.zeros(shape=(len(timestep_array), nos), dtype=np.complex)

    for idx, t in enumerate(timestep_array):
        psi_array[idx] = psi_t(psi_initial, hamiltonian, t)

    return psi_array, timestep_array
'''
