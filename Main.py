import itertools as it
import time
import math as mt

import numpy as np

import SubRoutine1
import SubRoutine2
import SubRoutine3
import Output

__author__ = "Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics"


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
        # Time Evolution - starting time, ending time and no. of time steps; time in seconds
        self.t_initial = 0.0
        self.t_final = 1.0
        self.t_steps = 10

        # No of  lattice sites eg. nsa = 3 => nol = 9
        self.nol = self.nsa ** 2
        # No. of sites in sub-lattice B
        self.nol_b = self.nol - (self.nol_a + len(self.lat_del_pos))
        # Site joining sub-lattice A and B (numbered after deleting sites)
        self.link_pos = mt.sqrt(self.nol_a) * self.nsa - (self.nsa - mt.sqrt(self.nol_a))
        # Lattice after deleting sites
        self.lat = np.arange(1, self.nol + 1, dtype=np.int32)
        # Time gap between successive steps
        self.delta_t = (self.t_final - self.t_initial) / self.t_steps


def eigenstates_lattice(lat, nop, lat_del_pos):
    # print 'lattice sites=', lat
    if np.size(lat_del_pos) != 0:
        lat_del = np.delete(lat, lat_del_pos - 1)
        eigenstates = np.array(list(it.combinations(lat_del, nop)), dtype=np.int32)
        # print 'lattice sites(after deletion)=', lat_del
    else:
        eigenstates = np.array(list(it.combinations(lat, nop)), dtype=np.int32)

    return eigenstates, len(eigenstates)


def main():
    s = System()

    eigenstates, nos = eigenstates_lattice(s.lat, s.nop, s.lat_del_pos)
    eigenstates_a, nos_a = eigenstates_lattice(s.lat, s.nop, s.lat_del_pos_a)
    np.savetxt('Output/Eigenstates.csv', eigenstates, delimiter=',', fmt='%1d')
    Output.status(1)

    # -----Sub-Routine 1 (Hamiltonian, Eigenvalues and Eigenvectors)-----

    # Hamiltonian
    h_time1 = time.time()
    hamiltonian = SubRoutine1.parallel_call_hamiltonian(eigenstates, nos, s.nsa, s.nop)
    hamiltonian_a = SubRoutine1.parallel_call_hamiltonian(eigenstates_a, nos_a, s.nsa, s.nop)
    h_time2 = time.time()
    np.savetxt('Output/Hamiltonian.csv', hamiltonian, delimiter=',', fmt='%1d')
    Output.status(2, h_time2 - h_time1)

    # Eigenvalues and Eigenvectors
    e_time1 = time.time()
    eigenvalues, eigenvectors = SubRoutine1.eigenvalvec(hamiltonian)
    eigenvalues_a, eigenvectors_a = SubRoutine1.eigenvalvec(hamiltonian_a)
    e_time2 = time.time()
    np.savetxt('Output/Eigenvalues.csv', eigenvalues, delimiter=',')
    np.savetxt('Output/Eigenvectors.csv', eigenvectors, delimiter=',')
    Output.status(3, e_time2 - e_time1)

    # ------Sub-Routine 2 (Recursion Time and State Relabelling)-----

    # Recursion Time
    # recur_time = SubRoutine2.recursion_time(eigenvalues)
    # Output.status(4, recur_time)

    # Relabelling
    r_time1 = time.time()
    relabelled_states = SubRoutine2.relabel(eigenstates)
    r_time2 = time.time()
    Output.status(5, r_time2 - r_time1)

    # -----Sub-Routine 3 (Time Evolution and Von-Neumann Entropy)-----

    # Time Evolution
    evo_time1 = time.time()
    psi_initial = SubRoutine3.random_eigenvector(eigenvectors_a, relabelled_states, nos, nos_a, s.nop)
    psi_t, timestep_array = SubRoutine3.time_evolution(psi_initial, hamiltonian, nos)
    evo_time2 = time.time()
    np.savetxt('Output/Psi.csv', psi_t, delimiter=',')
    Output.status(6, evo_time2 - evo_time1)

    # Von-Neumann Entropy
    vn_entropy_b = SubRoutine3.von_neumann_b(psi_t, relabelled_states, nos)
    np.savetxt('Output/Entropy_B.csv', vn_entropy_b, delimiter=',')

    # -----Output-----
    Output.plotting(timestep_array, np.real(vn_entropy_b))
    # Output.printout(nos, eigenstates, hamiltonian, eigenvalues, eigenvectors)
    # Output.plotting(np.arange(0, len(eigenvalues)), np.real(eigenvalues))

    # -----Terminate-----
    raise SystemExit


if __name__ == '__main__':
    main()
