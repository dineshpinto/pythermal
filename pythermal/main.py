# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Thermal equilibrium of hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group
# St. Stephen's Centre for Theoretical Physics, New Delhi
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import print_function, division, absolute_import

import os
from datetime import datetime
from multiprocessing import cpu_count
from time import time

if 'OPENBLAS_NUM_THREADS' not in os.environ:
    if cpu_count() >= 16:
        os.environ['OPENBLAS_NUM_THREADS'] = '16'
    elif 4 <= cpu_count() < 16:
        os.environ['OPENBLAS_NUM_THREADS'] = '{}'.format(cpu_count())
    else:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.linalg as la

from output import *
from routines import *


__author__ = "Thermalization and Quantum Entanglement Project Group, SSCTP"
__version__ = "v2.1.0"


class System:
    def __init__(self, initial_value, lattice_a=None, lattice_b=None):
        """
        Stores metadata about the system.
        :param initial_value: List of initial state values of the system
        :param lattice_a: Lattice sites in A (only used when manually
        defining lattice)
        :param lattice_b: Lattice sites in B (only used when manually
        defining lattice)
        """
        # No of particles
        self.nop = int(initial_value[0])

        # Dimensionality of lattice
        self.n_dim = int(initial_value[1])

        # Whole lattice
        self.lattice = np.concatenate((lattice_a, lattice_b))

        # No. of sites in sub-lattice A
        self.nol_a = len(lattice_a)

        # No. of sites in sub-lattice B
        self.nol_b = len(lattice_b)

        # Total no. of  lattice sites
        self.nol = len(self.lattice)

    @property
    def folder_path(self):
        """
        The naming convection is as follows:
        P[Total no. of particles]
        D[Dimensionality of lattice]
        A[No. of particles in A]
        B[No. of particles in B]

        :return: Path for storing program output
        """
        return ('../pythermal_output/P[{}]D[{}]A[{}]B[{}]/'
                .format(self.nop, self.n_dim, self.nol_a, self.nol_b))

    def check_system(self):
        """
        Runs checks to make sure all inputs are valid.

        :raises Value errors for invalid inputs
        """
        if not self.nop > 0:
            raise ValueError('No. of particles should be greater than 0')

        if not self.nop < self.n_dim ** 2:
            raise ValueError('Too many particles [{}] for lattice [{}]'
                             .format(self.nop, self.n_dim ** 2))

        if not self.n_dim > 3:
            raise ValueError('Shape of lattice must be at least 3')

    @staticmethod
    def plotting_metadata():
        """
          Stores metadata for MatPlotLib plots.

          :return: Filename of images
          :return: Image titles
          :return: y axis labels
          :return: x axis labels
          :return: y axis limits
          """
        image_names = ['VN_Entropy_B', 'Trace_sqr_B', 'Avg_A', 'Avg_B',
                       'Avg_AB']

        titles = [r'Von-Neumann entropy ($S_{VN}$) vs time ($\tau$)',
                  r'Purity ($tr(\rho^2))$) vs time ($\tau$)',
                  r'Avg. particles in A vs time ($\tau$)',
                  r'Avg. particles in B vs time ($\tau$)',
                  r'Avg. particles in A and B vs time ($\tau$)']

        y_labels = [r'Von-Neumann Entropy $[S_{VN} = - tr(\rho \ln(\rho))]$',
                    r'Purity $[tr(\rho^2))] \rightarrow$',
                    r'Avg. particles in A', r'Avg. particles in B',
                    r'Avg. particles in A and B']

        x_labels = [r'Time $[\tau]\rightarrow$']

        y_limits = [(0.0, 3.0), (-1.0, 2.0), (0.0, 5.0)]

        return image_names, titles, y_labels, x_labels, y_limits

    def check_existence(self, names):
        """
        Checks whether variables exists on hard disk.
        Generally deprecated for try-except statements.

        :param names: list of names of variables
        :return: Boolean list whether files exists on hard disk
        """
        existence = [False] * len(names)

        for idx, name in enumerate(names):
            if os.path.isfile(self.folder_path + name):
                existence[idx] = True
                print('{} exists at {}'.format(name, self.folder_path))
            else:
                print('{} does not exist'.format(name))
        print('')

        return existence


def main_states(initial_values, chosen_eigenstates, lattice_a, lattice_b):
    """
    Contains functions to find the density matrix of a subsystem in its
    energy basis and compare the diagonal/off-diagonal elements.
    Uses time() module to time execution of various functions and output
    to standard output.


    :param chosen_eigenstates: Eigenstates for which to compute DM's
    :param initial_values: List of initial values for system
        initial_values = [
            Total no. of particles(nop),
            Dimension of lattice(ndims)]
    :param lattice_a: List of sites in A
    :param lattice_b: List of sites in B
    :return: True if execution successful
    """
    s = System(initial_values, lattice_a, lattice_b)
    path = s.folder_path
    s.check_system()

    tot_time1 = time()

    # Position States
    try:
        pos_states_ab = read_file(path, 'PositionStates_AB.csv')
        nos_ab = len(pos_states_ab)
        pos_states_a = read_file(path, 'PositionStates_A.csv')
        nos_a = len(pos_states_a)
    except IOError:
        pos_states_ab, nos_ab = position_states(s.lattice, s.nop)
        pos_states_a, nos_a = position_states(lattice_a, s.nop)
        write_file(path, 'PositionStates_AB.csv', pos_states_ab, fmt='%1d')
        write_file(path, 'PositionStates_A.csv', pos_states_a, fmt='%1d')

    # Hamiltonian
    try:
        hamiltonian = read_file(path, 'Hamiltonian_AB.csv', dtype=int)
    except IOError:
        print("Hamiltonian...")
        hamiltonian = hamiltonian_parallel(s.lattice, s.n_dim, s.nop)
        write_file(path, 'Hamiltonian_AB.csv', hamiltonian, fmt='%1d')
        print("Hamiltonian done!")

    # Eigenvalues and Eigenvectors
    e_time1 = time()
    try:
        eigenvalues = read_file(path, 'Eigenvalues_AB.csv')
        eigenvectors = read_file(path, 'Eigenvectors_AB.csv', dtype=complex)
    except IOError:
        print("Eigenvalues/Eigenvectors of Hamiltonian...")
        eigenvalues, eigenvectors = diagonalize(hamiltonian)
        write_file(path, 'Eigenvalues_AB.csv', eigenvalues)
        write_file(path, 'Eigenvectors_AB.csv', eigenvectors)
        print("Eigenvalues/Eigenvectors of Hamiltonian done!")
    e_time2 = time()
    status(e_time2 - e_time1)

    # Block diagonal Hamiltonian
    bd_time1 = time()
    try:
        h_bd = read_file(path, 'Hamiltonian_B_BD.csv')
    except IOError:
        print("Block diagonal...")
        h_bd = h_block_diagonal(lattice_b, s.n_dim, s.nop)
        write_file(path, 'Hamiltonian_B_BD.csv', h_bd, fmt='%1d')
        print("Block diagonal done!")

    # Diagonalizing block diagonal Hamiltonian
    try:
        h_bd_evecs = read_file(path, 'Hamiltonian_B_BD_Evecs.csv',
                               dtype=complex)
    except IOError:
        print("Eigenvalues/Eigenvectors of block diagonal...")
        _, h_bd_evecs = diagonalize(h_bd)
        write_file(path, 'Hamiltonian_B_BD_Evecs.csv', h_bd_evecs)
        print("Eigenvalues/Eigenvectors of block diagonal done!")
    bd_time2 = time()
    status(bd_time2 - bd_time1)

    # State Relabelling
    print("Relabelling...")
    labels = relabel(pos_states_ab, s.nop, s.nol_b, lat_a=lattice_a)
    write_file(path, 'RelabelledStates.csv', labels)

    # Density Matrices for chosen eigenstates
    for state_num in chosen_eigenstates:
        print("Eigenstate = {}/{}".format(state_num, nos_ab))
        # Normalize state
        state = eigenvectors[state_num] / la.norm(eigenvectors[state_num])

        # Density Matrix
        dm_time1 = time()
        rho_fname = '[{}]{}.csv'.format(state_num, eigenvalues[state_num])

        # Density matrix in position basis
        try:
            rho_pbasis = read_file(path + 'RhoStates(PositionBasis)/',
                                   rho_fname, dtype=complex)
        except IOError:
            print("DM in position basis...")
            rho_pbasis = rho_b_pbasis(labels, state, nos_ab, s.nol_b,
                                      s.nop)
            rho_fname = ('[{}]{}.csv'.format(state_num,
                                             eigenvalues[state_num]))
            write_file(path + 'RhoStates(PositionBasis)/', rho_fname,
                       rho_pbasis)

        # Density matrix in energy basis
        try:
            rho_ebasis = read_file(path + 'RhoStates(EnergyBasis)/', rho_fname,
                                   dtype=complex)
        except IOError:
            print("Transforming DM to Energy basis...")
            rho_ebasis = transform_basis(rho_pbasis, h_bd_evecs)
            write_file(path + 'RhoStates(EnergyBasis)/', rho_fname, rho_ebasis)

        dm_time2 = time()
        status(dm_time2 - dm_time1)

        # Place tests for Density matrices after this

        # Naive check for thermal DM
        max_diag, max_offdiag = naive_thermal(rho_ebasis)
        print("Diagonal/Off-diagonal = {}/{}".format(max_diag, max_offdiag))

        # Append output data to log.txt
        with open(path + 'log.txt', mode='a') as f:
            string = ("{}\n{}"
                      "\nNo. of particles = {}"
                      "\nNo. of lattice sites in A = {}"
                      "\nNo. of lattice sites in B = {}"
                      "\nTotal no. of lattice sites = {}"
                      "\nEigenstate = {}/{}"
                      "\nMax. diagonal = {}"
                      "\nMax. off-diagonal = {}"
                      "\nRatio (Max diag/Max off-diag)= {}"
                      "\n"
                      .format('*' * 79, datetime.now(), s.nop, s.nol_a,
                              s.nol_b, s.nol, state_num, nos_ab, max_diag,
                              max_offdiag, max_diag / max_offdiag))
            f.write(string)

    print("Output log at {}".format(path + 'log.txt'))

    tot_time2 = time()
    status(tot_time2 - tot_time1)

    return True


def main_time(initial_values, chosen_eigenstate, t_initial, t_final, t_steps,
              lattice_a, lattice_b):
    """
    Contains functions to time evolve the entire system (both
    sub-lattices) starting in an initial state where all particles are in
    one sub-lattice. Possible issues.

    :param initial_values: List of initial values for system
        initial_values = [
            Total no. of particles(nop),
            Dimension of lattice(ndims)]
    :param chosen_eigenstate: Initial state for time evolution
    :param t_initial: Starting time
    :param t_final: Ending time
    :param t_steps: No. of steps
    :param lattice_a: List of sites in A
    :param lattice_b: List of sites in B
    :return: True if execution successful
    """
    s = System(initial_values, lattice_a, lattice_b)
    path = s.folder_path
    s.check_system()

    timesteps = np.arange(t_initial, t_final, (t_final - t_initial) / t_steps)

    tot_time1 = time()

    # Position States
    try:
        pos_states_ab = read_file(path, 'PositionStates_AB.csv')
        nos_ab = len(pos_states_ab)
        pos_states_a = read_file(path, 'PositionStates_A.csv')
        nos_a = len(pos_states_a)
    except IOError:
        pos_states_ab, nos_ab = position_states(s.lattice, s.nop)
        pos_states_a, nos_a = position_states(lattice_a, s.nop)
        write_file(path, 'PositionStates_AB.csv', pos_states_ab, fmt='%1d')
        write_file(path, 'PositionStates_A.csv', pos_states_a, fmt='%1d')

    # Hamiltonian
    try:
        hamiltonian = read_file(path, 'Hamiltonian_AB.csv')
    except IOError:
        print('Hamiltonian...')
        hamiltonian = hamiltonian_parallel(s.lattice, s.n_dim, s.nop)
        write_file(path, 'Hamiltonian_AB.csv', hamiltonian, fmt='%1d')

    try:
        hamiltonian_a = read_file(path, 'Hamiltonian_A.csv')
    except IOError:
        hamiltonian_a = hamiltonian_parallel(lattice_a, s.n_dim, s.nop)
        write_file(path, 'Hamiltonian_A.csv', hamiltonian_a, fmt='%1d')

    # Eigenvalues and Eigenvectors
    e_time1 = time()
    try:
        eigenvalues_ab = read_file(path, 'Eigenvalues_AB.csv')
        eigenvectors_ab = read_file(path, 'Eigenvectors_AB.csv', dtype=complex)
    except IOError:
        print('Diagonalizing...')
        eigenvalues_ab, eigenvectors_ab = diagonalize(hamiltonian)
        write_file(path, 'Eigenvalues_AB.csv', eigenvalues_ab)
        write_file(path, 'Eigenvectors_AB.csv', eigenvectors_ab)

    # Eigenvalues and Eigenvectors of A
    try:
        eigenvalues_a = read_file(path, 'Eigenvalues_A.csv')
        eigenvectors_a = read_file(path, 'Eigenvectors_A.csv', dtype=complex)
    except IOError:
        print('Eigenvalues/Eigenvectors of A...')
        eigenvalues_a, eigenvectors_a = diagonalize(hamiltonian_a)
        write_file(path, 'Eigenvalues_A.csv', eigenvalues_a)
        write_file(path, 'Eigenvectors_A.csv', eigenvectors_a)
        print('Eigenvalues/Eigenvectors of A done!')
    e_time2 = time()
    status(e_time2 - e_time1)

    # State Relabelling
    labels = relabel(pos_states_ab, s.nop, s.nol_b, lat_a=lattice_a)
    write_file(path, 'RelabelledStates.csv', labels)

    # Initial state as eigenvector of A
    psi_initial = initial_sublattice_state(eigenvectors_a, labels, nos_ab,
                                           s.nop, chosen_eigenstate)

    # Used to create a unique filename for each set of times
    t_data = '[{},{},{}]'.format(t_initial, t_final, t_steps)

    # Time Evolution
    evo_time1 = time()
    try:
        psi_t = read_file(path, 'Psi_t{}.csv'.format(t_data), dtype=complex)
    except IOError:
        print('Time evolving...')
        psi_t = time_evolution(psi_initial, hamiltonian, nos_ab, timesteps)
        write_file(path, 'Psi_t{}.csv'.format(t_data), psi_t)
        print('Time evolution done!')
    evo_time2 = time()
    status(evo_time2 - evo_time1)

    # Average number of particles in A and B
    try:
        avg_a = read_file(path, 'Avg_A{}.csv'.format(t_data))
        avg_b = read_file(path, 'Avg_B{}.csv'.format(t_data))
    except IOError:
        avg_a, avg_b = avg_particles(psi_t, timesteps, labels, s.nop)
        write_file(path, 'Avg_A{}.csv'.format(t_data), avg_a)
        write_file(path, 'Avg_B{}.csv'.format(t_data), avg_b)

    # Von-Neumann Entropy
    vn_time1 = time()
    try:
        entropy_b = read_file(path, 'Entropy_B{}.csv'
                              .format(t_data), dtype=complex)
        tr_sqr_b = read_file(path, 'Trace_Square_B{}.csv'.format(t_data))
    except IOError:
        print('Von-Neumann Entropy...')
        entropy_b, tr_sqr_b = vn_entropy_b(psi_t, labels, nos_ab, s.nol_b,
                                           s.nop)
        write_file(path, 'Entropy_B{}.csv'.format(t_data), entropy_b)
        write_file(path, 'Trace_Square_B{}.csv'.format(t_data), tr_sqr_b)
        print('Von-Neumann Entropy done!')
    vn_time2 = time()
    status(vn_time2 - vn_time1)

    # Generate images
    show_images = False
    image_name, titles, y_labels, x_labels, y_limits = s.plotting_metadata()
    path_images = path + 'TimeEvolutionImages{}/'.format(t_data)

    plot_write(timesteps, entropy_b, titles[0], y_labels[0], x_labels[0],
               y_limits[0], path_images, image_name[0], show_images)
    plot_write(timesteps, tr_sqr_b, titles[1], y_labels[1], x_labels[0],
               y_limits[0], path_images, image_name[1], show_images)
    plot_write(timesteps, avg_a, titles[2], y_labels[2], x_labels[0],
               y_limits[2], path_images, image_name[2], show_images)
    plot_write(timesteps, avg_b, titles[3], y_labels[3], x_labels[0],
               y_limits[2], path_images, image_name[3], show_images)
    plot_write(timesteps, avg_b + avg_a, titles[4], y_labels[4], x_labels[0],
               y_limits[2], path_images, image_name[4], show_images)

    tot_time2 = time()
    status(tot_time2 - tot_time1)
    return True


if __name__ == '__main__':
    """
    init_values = [Total no. of particles(nop), Dimension of lattice(ndims)]
    chosen_estate = [Eigenstates  for which to compute DM's]
    """
    init_values = [2, 6]
    chosen_e_states = [x for x in range(10, 20, 2)]
    sub_lattice_a = np.genfromtxt('a.txt', dtype=int)
    sub_lattice_b = np.genfromtxt('b.txt', dtype=int)

    # main_states(init_values, chosen_e_states, sub_lattice_a, sub_lattice_b)
    main_time(init_values, 0, 0, 100, 20, sub_lattice_a, sub_lattice_b)
