# PyThermal - Time evolving fermions on a two-dimensional crystal lattice
Program to simulate n-particles on a 2 dimensional lattice, which is divided into sub-lattices A and B after
deletion of sites. The variation of the Von-Neumann entropy of these sub-lattices is then studied.

## Task List 

-[ ] recursion_time overflow error if no. of inputs >= 50
-[ ] Complete documentation 

## Changelog (02-10-2015)

1. recursion_time(), gcd(), lcm(), lcm_call() added
2. Complete program structure redesign
2. class System created with __init__ to store variables(defined in documentation)
3. Extensive documentation added 



## Program Structure

Documentation by D. Pinto

### class System (function __init__)

        Parameters:
            nop: total no. of particles, int
            nsa: shape of square lattice, int
            nol_a: no. of sites in sub-lattice A, int
            lat_del_pos: positions of deleted sites, int, optional
            log_choice: export to LOG.txt, bool, optional
        Generated automatically:
            nol: total no. of lattice sites, int
            nol_b: no. of sites in sub-lattice B, int
            link_pos: site joining sub-lattices A and B, int
            lat: lattice sites array, np.int32

### main()

        eigenstates_lat(lat, nop, lat_del_pos)
                Parameters:
                    lat: lattice sites array, np.int32
                    nop: no. of particles, int
                    lat_del_pos: array of positions to delete, np.int32, optional
                Returns:
                    e_states: array of eigenstates, np.int32
                    len(e_states): total no. of eigenstates, int

#### - Sub-Routine 1

        hamiltonian_2d(start, stop, nos, nsa, nop, eigenstates, queue, h)
                Parameters:
                        start: start point of interator [j], int
                        stop: end point of iterator [j], int
                        nsa: shape of square lattice, int
                        nop: total no. of particles, int
                        eigenstates: array of eigenstates, np.int32
                        queue: mutiprocessing queue to store each processes' output
                        h: hamiltonian array, np.float32
                        
        distribute(n_items, n_processes, i)
                Parameters:
                        n_items: total number of items to compute, int
                        n_processes: no. of processes to create (= no. of cores), int
                        i: iterates over no. of processes, int
                Returns:
                        start: start point of interator [j], int
                        stop: end point of iterator [j], int
                        
        parallel_call_hamiltonian(e_states, nos, nsa, nop)
                Parameters:
                    e_states: array of eigenstates, np.int32
                    nos: total no. of states, int
                    nsa: shape of 2D lattice, int
                    nop: total no. of particles, int
                Returns:
                    h: 2D hamiltonian array, np.float32
                Environment:
                    C: Numpy
                    Python: Multiprocessing
                    
        eigenvalvec(h)
                Parameters:
                    h: 2D hamiltonian array, np.float32
                Returns:
                    e_vecs: eigenvector array, complex
                    e_vals: eigenvalue array, complex
                Environment:
                    Fortran: OpenBLAS, OpenMP

#### - Sub-Routine 2

        recursion_time()[alpha] 
        
        relabel(e_states, nol_a, nol_b, link_pos, nop)
                Parameters:
                    e_states: array of eigenstates, np.int32
                    nol_a: no. of sites in sub-lattice A, int
                    nol_b: no. of particles in sub-lattice B, int
                    link_pos: site joining sub-lattices A and B, int
                    nop: total no. of particles, int
                Returns:
                    np.array(y): array containing relabelled states, np.float64
                Environment:
                    Python
                    
        denmatrix_a(label, e_vecs, nos, nop, nol_a)
                nol_b replaced by nol_a
                See denmatrix_b below
        
        denmatrix_b(label, e_vec, nos, nop, nol_b)
                Parameters:
                    label: array containing relabelled states,
                    e_vecs: eigenvector array, complex
                    nop: total no. of particles, int
                    nol_b: no. of particles in sub-lattice B, int
                Returns:
                    density_mat: 2D density matrix for sub-lattice B, complex
                    den_trace: Sum of diagonal, complex
                    den_trace2: Sum of diagonal of product of DM with its conjugate, complex
                Environment:
                    Fortran: OpenBLAS, OpenMP

#### - Sub-Routine 3

        random_eigenvector(eigenvectors, nos)(eigenvectors, relabelled_states, nos, nos_a, nop)
                Parameters:
                    eigenvectors: eigenvector array, np.complex
                    relabelled_states: 
                    nos: total no. of states, int
                    nos_a: 
                    nop: 
                Returns:
                    psi_initial: randomly chosen eigenvector used as initial state, np.complex
                Environment:
                    C: Numpy
                    Python
                    
        von_neumann_b(psi_array, labels, nos)
                Parameters:
                        psi_array: 
                        labels: array containing relabelled states, np.float64
                        nos:
                Returns:
                        entropy_b: array containing Von-Neumann entropies, np.complex
                Environment:
                        C: Numpy
                        Fortran: OpenBLAS, OpenMP
                        
        psi_t(eigenvectors, eigenvalues, nos, psi_initial, t)
                Parameters:
                        eigenvectors: eigenvector array, np.complex
                        eigenvalues:
                        nos:
                        psi_initial: initial state, np.complex
                        t: time, float
                Returns:
                        psi: array containing psi at time t 
                        
        time_evolution(eigenvectors, eigenvalues, psi_initial, nos)
                Parameters:
                        eigenvectors: 
                        eigenvalues: 
                        psi_initial: 
                        nos:
                Returns:
                        psi_array: array of arrays containing psi at t, t + dt ...
 

## Previous build(s)

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
