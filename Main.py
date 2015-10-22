import itertools as it
import time
import math as mt
import numpy as np
# import fractions

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
        self.t_final = 200
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

    # --Sub-Routine 2--

    # Recursion Time
    print SubRoutine2.recursion_time(1, eigenvalues)
    exit()

    # Relabelling
    r_time1 = time.time()

    labels = SubRoutine2.relabel(eigenstates, s.nol_b, s.link_pos, s.nop)

    r_time2 = time.time()
    Output.status_output(4, r_time2 - r_time1)

    # --Sub-Routine 3--

    # Time Evolution
    evo_time1 = time.time()
    # psi_initial = SubRoutine3.random_eigenvector(eigenvectors, labels, nos, nos_a, s.nop)
    psi_initial = SubRoutine3.random_eigenvector(eigenvectors_a, labels, nos, nos_a, s.nop)

    # 1. Using psi(t) = psi(0)*exp(-i*H*t)
    # psi_array, timestep_array = SubRoutine3.time_evolution(psi_initial, hamiltonian, nos)

    # 2. Using psi(t) = SIGMA[|E_i><E_i|psi(0)>exp(-i*E_i*t / hbar)]
    psi_array, timestep_array = SubRoutine3.time_evolution(eigenvectors, eigenvalues, psi_initial, nos)

    evo_time2 = time.time()
    Output.status_output(5, evo_time2 - evo_time1)

    # Von-Neumann Entropy
    vn_entropy_b = SubRoutine3.von_neumann_b(psi_array, labels, nos)

    # --Output--

    # Output.printout(nos, eigenstates, hamiltonian, eigenvalues, eigenvectors)
    Output.plotting(timestep_array, np.real(vn_entropy_b))
    Output.plotting(np.arange(0, len(eigenvalues)), np.real(eigenvalues))

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

Previous build(s)
1. Function eigenstates() rewritten to account for missing lattice sites, site deletion controlled by [lat_del_pos]
2. Function nos() deprecated, nos replaced with len(c) in eigenstates()
3. Changed Hamiltonian, using if conditions to place 1's(on numpy.zeros matrix)
4. Parallelization of hamiltonian() governed by distribution function distribute()
5. P. carried out using Process function from multiprocessing library
6. multiprocessing.queue to store output of each process and clear(optional, improves stability) it afterwards
7. Replaced math.fabs() with abs()
8. Added elif and else statements
9. la.eig replaced with la.eigh to exploit symmetry of Hamiltonian matrix
10. Original (la.eig)eigenvalvec() deprecated, replaced with (la.eigh)eigenvalvec()
11. Separate timers for hamiltonian() and eigenvalvec()
12. Sizes of all arrays printed
13. Output printed to file("LOG.txt") using tabulate
14. OpenBLAS(/opt/OpenBLAS) linkage of Numpy(and consequently Scipy) in virtualenv "pyenv"
15. Program down-dated to work with Python 2.x
16. Functions ncr(), sum_ncr(), relabel() and denmatrix() added
17. la.eigh deprecated, la.eig reinstated to generate complex eigenvectors
18. Hamiltonian np.zeros switched to np.float32 data type
19. else condition(in hamiltonian_2d) now with continue statements
20. Direct call to hamiltonian_2d deprecated
21. Output made more verbose
22. Function ncr() deprecated
'''
