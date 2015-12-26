# PyThermal - Time evolving fermions on a two-dimensional crystal lattice

**Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics**

*Project Mentor: Dr. A. Gupta*   
*Project Students: A. Kumar, D. Pinto and M. Ghosh*

Program to simulate n-particles on a 2 dimensional lattice, which is divided into sub-lattices A and B after
deletion of sites. The variation of the Von-Neumann entropy of these sub-lattices is then studied.

## Task List 
- [ ] Evolve with a few eigenstates (estimate recurrence using ~4 decimal places)
- [ ] *SubRoutine2.recursion_time()* Recursion time calculation using LCM of the inverse of energy eigenvalues [Beta]


## Changelog (16-12-2015) 
+ *eigenstates()* function shifted to *SubRoutine1*
+ Recursion time calculation added [Beta]
+ Full compatibility with both Python 2 and 3
+ Error checking now outputs to stderr


## Program Structure 

*Documentation by D. Pinto*

The code is centered around the main function, from which the entire program can be controlled. It derives data from a 
class which is used to store initial values. The main function calls are subdivided into three sets of routines termed 
*Sub-Routines* and an Output/Plotting function, all of which are stored in separate source files.

### Note
The code was designed on Python 2.7 and will not work with versions older than Python 2.6. It is fully compatible 
with Python 3.x (no modifications necessary).   

This code requires the following header files:
1. Numpy/Scipy: Build against Fortran OpenBLAS for parallel processing using OpenMP
2. MatPlotLib 
3. Multiprocessing
4. Tqdm (optional)


### class System 
        
    __init__()
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

#### - Sub-Routine 1 (Hamiltonian, Eigenvalues and Eigenvectors)

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

#### - Sub-Routine 2 (Relabelling, Density Matrices and Recursion Time)

    recursion_time()[in development] 
    
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

#### - Sub-Routine 3 (Time Evolution and Von-Neumann Entropy)

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
 
### - Output/Plotting 

    status_output()
        Returns program status along with run times, requires header files from humanize

    printout() 
        Extensive output function, class option to export certain output to LOG.txt, requires 
        tabulate header files
                
    plotting()
        Generalized plotting for 2D graphs, uses MatPlotLib
    

## Previous build(s)

### Changelog (21/1/2015)
1. Function eigenstates() rewritten to account for missing lattice sites, site deletion controlled by [lat_del_pos]
2. Function nos() deprecated, nos replaced with len(c) in eigenstates()
3. Changed Hamiltonian, using if conditions to place 1's(on numpy.zeros matrix)


### Changelog (15/2/2015)

4. Parallelization of hamiltonian() governed by distribution function distribute()
5. Parallel processing carried out using Process function from multiprocessing library
6. multiprocessing.queue to store output of each process and clear(optional, improves stability) it afterwards

### Changelog (1/3/2015)

7. Replaced math.fabs() with abs()
8. Added elif and else statements


### Changelog (16/4/2015)

9. la.eig replaced with la.eigh to exploit symmetry of Hamiltonian matrix
10. Original (la.eig)eigenvalvec() deprecated, replaced with (la.eigh)eigenvalvec()


### Changelog (23/15/2015)

11. Separate timers for hamiltonian() and eigenvalvec()
12. Sizes of all arrays printed
13. Output printed to file("LOG.txt") using tabulate

### Changelog (26/6/2015)

14. OpenBLAS(/opt/OpenBLAS) linkage of Numpy(and consequently Scipy) in virtualenv "pyenv" [for SSCTP workstation]
15. Program down-dated to work with Python 2.x
16. Functions ncr(), sum_ncr(), relabel() and denmatrix() added
17. la.eigh deprecated, la.eig reinstated to generate complex eigenvectors
18. Hamiltonian np.zeros switched to np.float32 data type
19. else condition(in hamiltonian_2d) now with continue statements

### Changelog (6/7/2015)

20. Direct call to hamiltonian_2d deprecated
21. Output made more verbose
22. Function ncr() deprecated

### Changelog (19/8/2015)

23. Parallel processing of Von-Neumann entropy calculation and time-evolution (using OpenBLAS linkage with OpenMP for multiple threads sidesteps Python GIL)
24. Trace of density matrix B = 1.95 (almost constant, only observed under 1D time evolution), instead of 1.00

### Changelog (13/9/2015)

25. Recursion time temporary fix using inverse of |least eigenvalue|
26. recursion_time(), gcd(), lcm(), lcm_call() added

### Changelog (1/10/2015)

27. Complete program structure redesign
28. class System created to store variables(defined in documentation)
29. Extensive documentation added 

### Changelog (15/12/2015)

30. Error checking for trace of density matrix (Permitted error = 1.0e-4) 
31. Write output to disk (.csv) during program run
32. Large eigenvalue problem fixed (within Main.py)

### Changelog (16/12/2015)
33. tqdm reinstated for measuring progress of time-evolution and entropy [loop counter tqdm.tqdm]
34. Error checking for trace of density matrix made non-fatal [program execution uninterrupted]
35. Verbose output to disk
