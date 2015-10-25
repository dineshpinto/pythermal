import itertools as it
import time
import math as mt
import numpy as np

import SubRoutine1
import SubRoutine2
import SubRoutine3
import Output

__author__ = 'SSCTP - Entanglement Group'


class System:

    def __init__(self):
        # No of particles
        self.nop = 3
        # Shape of square 2D array i.e. 2x2, 3x3
        self.nsa = 4
        # Lattice sites to delete
        self.lat_del_pos = np.array([3, 4, 9, 13])  # [4, 5, 9, 10, 16, 17, 21, 22])
        # No. of sites in sub-lattice A
        self.nol_a = 4
        # Lattice A sites
        self.lat_del_pos_a = np.array([3, 4, 9, 13, 7, 8, 10, 11, 12, 14, 15, 16])
        # Time Evolution
        self.t_initial = 0.0
        self.t_final = 50
        self.t_steps = 100

        # No of  lattice sites eg. nsa = 3 => nol = 9
        self.nol = self.nsa ** 2
        # No. of sites in sub-lattice B
        self.nol_b = self.nol - (self.nol_a + len(self.lat_del_pos))
        # Site joining sub-lattice A and B (numbered after deleting sites)
        self.link_pos = mt.sqrt(self.nol_a) * self.nsa - (self.nsa - mt.sqrt(self.nol_a))
        # Lattice after deleting sites
        self.lat = np.arange(1, self.nol + 1, dtype=np.int32)


def eigenstates_lattice(lat, nop, lat_del_pos):
    print 'lattice sites=', lat
    if np.size(lat_del_pos) != 0:
        lat_del = np.delete(lat, lat_del_pos - 1)
        print 'lattice sites(after deletion)=', lat_del
        e_states = np.array(list(it.combinations(lat_del, nop)), dtype=np.int32)
    else:
        e_states = np.array(list(it.combinations(lat, nop)), dtype=np.int32)

    return e_states, len(e_states)


def main():
    s = System()

    eigenstates, nos = eigenstates_lattice(s.lat, s.nop, s.lat_del_pos)
    eigenstates_a, nos_a = eigenstates_lattice(s.lat, s.nop, s.lat_del_pos_a)
    Output.status_output(1)

    # --Sub-Routine 1--

    # Hamiltonian
    h_time1 = time.time()

    hamiltonian = SubRoutine1.parallel_call_hamiltonian(eigenstates, nos, s.nsa, s.nop)
    hamiltonian_a = SubRoutine1.parallel_call_hamiltonian(eigenstates_a, nos_a, s.nsa, s.nop)

    h_time2 = time.time()
    Output.status_output(2, h_time2 - h_time1)

    # Eigenvalues and Eigenvectors
    e_time1 = time.time()

    eigenvalues, eigenvectors = SubRoutine1.eigenvalvec(hamiltonian)
    eigenvalues_a, eigenvectors_a = SubRoutine1.eigenvalvec(hamiltonian_a)

    e_time2 = time.time()
    Output.status_output(3, e_time2 - e_time1)

    # Sorting the eigenvalues and eigenvectors in order of increasing eigenvalues
    idx = eigenvalues_a.argsort()
    eigenvalues_a = eigenvalues_a[idx]
    eigenvectors_a = eigenvectors_a[:, idx]

    # --Sub-Routine 2--

    # Recursion Time
    # print SubRoutine2.recursion_time(1, eigenvalues)

    # Relabelling
    r_time1 = time.time()

    relabelled_states = SubRoutine2.relabel(eigenstates, s.nol_b, s.link_pos, s.nop)

    r_time2 = time.time()
    Output.status_output(4, r_time2 - r_time1)

    # --Sub-Routine 3--

    # Time Evolution
    evo_time1 = time.time()
    # psi_initial = SubRoutine3.random_eigenvector(eigenvectors, relabelled_states, nos, nos_a, s.nop)
    psi_initial = SubRoutine3.random_eigenvector(eigenvectors_a, relabelled_states, nos, nos_a, s.nop)

    # 1. Using psi(t) = psi(0)*exp(-i*H*t)
    # psi_array, timestep_array = SubRoutine3.time_evolution(psi_initial, hamiltonian, nos)

    # 2. Using psi(t) = SIGMA[|E_i><E_i|psi(0)>exp(-i*E_i*t / hbar)]
    psi_array, timestep_array = SubRoutine3.time_evolution(eigenvectors, eigenvalues, psi_initial, nos)

    evo_time2 = time.time()
    Output.status_output(5, evo_time2 - evo_time1)

    # Von-Neumann Entropy
    vn_entropy_b = SubRoutine3.von_neumann_b(psi_array, relabelled_states, nos)

    # --Output--

    # Output.printout(nos, eigenstates, hamiltonian, eigenvalues, eigenvectors)
    Output.plotting(timestep_array, np.real(vn_entropy_b))
    # Output.plotting(np.arange(0, len(eigenvalues)), np.real(eigenvalues))

    # --Terminate--
    raise SystemExit


if __name__ == '__main__':
    main()


'''
NOTE:
Split runtime into n parts with (start, stop)
Divide into n processes each with a start, stop
Pass eigenvector, hamiltonian, start, stop to time evolution function -> get psi
Pass psi to density matrix to get density matrix
Find VN entropy by passing density matrix of b and log(density matrix of b)
Create large array VN array to store the all the entropies
'''
