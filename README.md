# PyThermal - Time evolving fermions on a 2D crystal lattice


CHANGELOG(02-10-2015)
1. Complete program structure redesign
2. class System created with __init__ to store variables(defined in documentation)
3. Semi-extensive documentation added (to be completed)


Program to simulate n-particles on a 2 dimensional lattice, which is divided into sub-lattices A and B after
deletion of sites. The variation of the Von-Neumann entropy of these sub-lattices is then studied.

The program structure is as follows:

    - class System, function __init__()
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

    - main()
        - Sub-Routine 1
            eigenstates_lat(lat, nop, lat_del_pos)
                Parameters:
                    lat: lattice sites array, np.int32
                    nop: no. of particles, int
                    lat_del_pos: array of positions to delete, np.int32, optional
                Returns:
                    e_states: array of eigenstates, np.int32
                    len(e_states): total no. of eigenstates, int

        - Sub-Routine 2
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

        - Sub-Routine 3
            eigenvalvec(h)
                Parameters:
                    h: 2D hamiltonian array, np.float32
                Returns:
                    e_vecs: eigenvector array, complex
                    e_vals: eigenvalue array, complex
                Environment:
                    Scipy -> Fortran: OpenBLAS, OpenMP

        - Sub-Routine 4
            random_eigenvector(e_vecs, nos)
                Parameters:
                    e_vecs: eigenvector array, complex
                    nos: total no. of states, int
                Returns:
                    e_vecs[rand]: randomly chosen eigenvector
                Environment:
                    Python

            relabel(e_states, nol_a, nol_b, link_pos, nop)
                Parameters:
                    e_states: array of eigenstates, np.int32
                    nol_a: no. of sites in sub-lattice A, int
                    nol_b: no. of particles in sub-lattice B, int
                    link_pos: site joining sub-lattices A and B, int
                    nop: total no. of particles, int
                Returns:
                    np.array(y): array containing relabelled states,
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
                    Scipy -> Fortran: OpenBLAS, OpenMP

