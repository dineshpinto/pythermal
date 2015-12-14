import numpy as np
import scipy.linalg as la

import SubRoutine2
import Main


# Returns a random eigenvector
def random_eigenvector(eigenvectors, relabelled_states, nos, nos_a, nop):
    psi_initial = np.zeros(nos, dtype=np.complex)
    rand = np.random.randint(0, nos_a)
    j = 0
    for i in relabelled_states[1]:
        if i == nop:
            psi_initial[i] = eigenvectors[rand][j]
            j += 1

    # print "\nEigenvector", rand, "chosen randomly for time evolution"
    a = la.norm(psi_initial)
    psi_initial /= a
    return psi_initial


# Calculates Von-Neumann entropy as entropy = - rho * ln(rho)
def von_neumann_b(psi_array, relabelled_states, nos):
    entropy_b = np.zeros(len(psi_array), dtype=np.complex)

    for idx, psi_val in enumerate(psi_array):
        d_matrix_b = SubRoutine2.denmatrix_b(relabelled_states, psi_val, nos)
        entropy_b[idx] = -1.0 * np.trace(np.dot(d_matrix_b, la.logm(d_matrix_b)))

    return entropy_b


# Psi evolved as psi(t) = psi(0) * exp(i * H * t)
def time_evolution(psi_initial, hamiltonian, nos):
    s = Main.System()

    timestep_array = np.arange(s.t_initial, s.t_final, s.delta_t)
    psi_array = np.zeros(shape=(len(timestep_array), nos), dtype=np.complex)

    for idx, t in enumerate(timestep_array):
        psi_array[idx] = np.dot(psi_initial, la.expm(-1.0j * 0.61 * 10**(-38) * hamiltonian * t))

    return psi_array, timestep_array


'''
def time_evolution(psi_initial, hamiltonian, nos):
    s = Main.System()

    timestep_array = np.arange(s.t_initial, s.t_final, s.delta_t)
    psi_array = np.zeros(shape=(len(timestep_array), nos), dtype=np.complex)
    a = np.zeros_like(hamiltonian)

    for idx, t in enumerate(timestep_array):
        for i in range(len(hamiltonian)):
            for j in range(len(hamiltonian)):
                a[i][j] = np.exp(-1.0j * hamiltonian[i][j] * t)

        psi_array[idx] = np.dot(psi_initial, a)

    return psi_array, timestep_array
'''





'''
# Psi evolved in accordance with 1D paper psi(t) = SIGMA_(i=0)^(n) [|E_i><E_i|psi(0)>exp(-i*E_i*t / hbar)]

def psi_t(eigenvectors, eigenvalues, nos, psi_initial, t):
    psi = np.zeros(nos, dtype=np.complex)
    for i in xrange(nos):
        psi += np.exp(-1.0j * eigenvalues[i] * t) * eigenvectors[:, i] * np.vdot(eigenvectors[:, i], psi_initial)

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
