# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Time evolving hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group
# St. Stephen's Centre for Theoretical Physics, New Delhi
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import print_function, division, absolute_import

import os
from collections import namedtuple
from time import time
import traceback

if 'OPENBLAS_NUM_THREADS' not in os.environ:
    os.environ['OPENBLAS_NUM_THREADS'] = '16'

import numpy as np
import scipy.linalg as la

from output import status, write_file, read_file, plotting
from routines import (position_states, hamiltonian_parallel, eig, relabel,
                      initial_state, vn_entropy, time_evolution,
                      avg_particles, h_block_diagonal, state_initializer,
                      density_matrix_b, transformation)

__author__ = "Thermalization and Quantum Entanglement Project Group, SSCTP"
__version__ = "v1.5.0"


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

        # Time Evolution - starting time, ending time and no. of time steps
        self.t_initial = float(initial_value[2])
        self.t_final = float(initial_value[3])
        self.t_steps = int(initial_value[4])

        # Eigenvector chosen
        self.state_num = int(initial_value[5])

        # No. of sites in sub-lattice A
        self.nol_a = len(lattice_a)

        # No. of sites in sub-lattice B
        self.nol_b = len(lattice_b)

        # No of  lattice sites before deletion eg. nsa = 3 => nol = 9
        self.nol = len(lattice_a) + len(lattice_b)

        # Lattice before deleting sites
        self.lattice = np.concatenate((lattice_a, lattice_b))

    @property
    def timesteps(self):
        """
        :return: Array of times at which the system will be sampled

        """
        # Time gap between successive time steps
        delta_t = (self.t_final - self.t_initial) / self.t_steps

        # Array storing various times
        timestep_array = np.arange(self.t_initial, self.t_final, delta_t)

        return timestep_array

    @property
    def fold_path_ti(self):
        """
        :return: Path for storing time independent variables
        """
        return ('../pythermal_output/TI-{}_{}_{}/'
                .format(self.nop, self.n_dim, self.nol_a))

    @property
    def fold_path_td(self):
        """
        :return: Path for storing time dependent variables
        """
        return ('../pythermal_output/TD-{}_{}_{}_{}/'
                .format(self.nop, self.n_dim, self.nol_a, self.t_final))

    @staticmethod
    def file_names():
        """
        :return: Names of variables for storing on hard disk

        """
        names = ['Hamiltonian_AB.csv', 'Hamiltonian_A.csv',
                 'Eigenvalues_AB.csv', 'Eigenvectors_AB.csv',
                 'Eigenvalues_A.csv', 'Eigenvectors_A.csv',
                 'Psi_t.csv', 'Avg_A.csv', 'Avg_B.csv',
                 'VN_Entropy_B.csv', 'VN_Trace2_B']

        return names

    @property
    def check_existence(self):
        """
        Checks whether variables exists on hard disk.
        :return: Integer list whether files exists on hard disk(1) or not(0)

        """
        names = self.file_names()
        existence = [False] * len(names)

        for idx, name in enumerate(names):

            if os.path.isfile(self.fold_path_ti + name):
                existence[idx] = True
                print('{} exists at {}'.format(name, self.fold_path_ti))

            elif os.path.isfile(self.fold_path_td + name):
                existence[idx] = True
                print('{} exists at {}'.format(name, self.fold_path_td))

            else:
                print('{} does not exist'.format(name))
        print('')

        return existence

    def check_system(self, options, lattice_a, lattice_b):
        """
        Runs checks to make sure all inputs are valid.
        Raises ValueError if not.
        :param options:
        :param lattice_a:
        :param lattice_b:

        """
        if lattice_a is None or lattice_b is None:
            raise ValueError('Enter both lattice A and lattice B')

        if not self.nop > 0:
            raise ValueError('No. of particles should be greater than 0')
    
        if self.nop > self.nol_a and not options[1]:
            raise ValueError('Too many particles [{}] for sub-lattice A [{}]'
                             .format(self.nop, self.nol_a))
    
        if not self.nop < self.n_dim ** 2:
            raise ValueError('Too many particles [{}] for lattice [{}]'
                             .format(self.nop, self.n_dim ** 2))
    
        if not self.n_dim > 3:
            raise ValueError('Shape of lattice must be at least 3')
    
        if not self.t_final >= self.t_initial:
            raise ValueError('Final time [{}] not more than initial time [{}]'
                             .format(self.t_final, self.t_initial))
    
        if not self.t_steps > 0:
            raise ValueError('No. of time steps has to be greater than 0')


def main_states(initial_values, options=None, lattice_a=None, lattice_b=None):
    """
    Contains calls to/control of all functions in program.

    :param options: List of options for program execution
        options = [
            Show images(1=YES),
            Check for files on disk(1=YES)]

    :param initial_values: List of initial values for system
        initial_values = [
            Total no. of particles(nop),
            Dimension of lattice(ndims),
            t_initial,
            t_final,
            t_steps,
            initial eigenvector no.,
            initial psi(1=eigenstate of entire system)]

    :param lattice_a: List of sites in A
    :param lattice_b: List of sites in B
    :return: True if execution successful

    """
    s = System(initial_values, lattice_a, lattice_b)
    path_ti, path_td, names = s.fold_path_ti, s.fold_path_td, s.file_names()
    s.check_system(options, lattice_a, lattice_b)

    tot_time1 = time()

    # Eigenstates
    pos_states_ab, nos_ab = position_states(s.lattice, s.nop)
    pos_states_a, nos_a = position_states(lattice_a, s.nop)
    status(1)
    write_file(path_ti, 'PositionStates_AB.csv', pos_states_ab, fmt='%1d')
    write_file(path_ti, 'PositionStates_A.csv', pos_states_a, fmt='%1d')

    print("Hamiltonian...")

    # Hamiltonian
    h_time1 = time()
    # hamiltonian = read_file(path_ti, names[0], dtype=int)
    hamiltonian = hamiltonian_parallel(s.lattice, s.n_dim, s.nop)
    write_file(path_ti, names[0], hamiltonian, fmt='%1d')
    h_time2 = time()
    status(2, h_time2 - h_time1)

    print ("Hamiltonian done!")

    print("Diagonalization...")

    # Eigenvalues and Eigenvectors
    e_time1 = time()
    # eigenvalues = read_file(path_ti, names[2])
    # eigenvectors = read_file(path_ti, names[3], dtype=complex)
    eigenvalues, eigenvectors = eig(hamiltonian)
    write_file(path_ti, names[2], eigenvalues)
    write_file(path_ti, names[3], eigenvectors)
    e_time2 = time()
    status(3, e_time2 - e_time1)

    print("Diagonalization done!")

    print("Block diagonal...")

    try:
        ham = []
        for l in range(s.nop + 1):
            ham.append(hamiltonian_parallel(lat_b, s.n_dim, l))
    except Exception as e:
        print(e, traceback.format_exc())
        ham = []
        h_inter = h_block_diagonal(lattice_b, s.n_dim)
        for h, k in zip(h_inter, range(s.nop + 1)):
            write_file(path_ti, 'Hamiltonian_B_{}.csv'.format(k), h,
                       fmt='%1d')
        for k in range(s.nop + 1):
            ham.append(read_file(path_ti, 'Hamiltonian_B_{}.csv'.format(k)))

    h_bd = la.block_diag(*ham)
    write_file(path_ti, 'Hamiltonian_B_BD_abnormal.csv', h_bd, fmt='%1d')

    print("Block diagonal done!")

    if np.allclose(np.transpose(h_bd), h_bd):
        print("Block diagonal is symmetric")
    else:
        print("WARNING: Block diagonal is not symmetric")

    print ("Diagonalizing block diagonal...")
    _, h_bd_evecs = eig(h_bd)

    print("Relabelling...")

    # State Relabelling
    r_time1 = time()
    labels = relabel(pos_states_ab, s.nop, s.nol_b, lat_a=lattice_a)
    r_time2 = time()
    status(4, r_time2 - r_time1)
    write_file(path_ti, 'RelabelledStates.csv', labels)

    print ("Chosen eigenstate = {}".format(s.state_num))
    state = eigenvectors[s.state_num] / la.norm(eigenvectors[s.state_num])
    rho_pbasis = density_matrix_b(labels, state, nos_ab, s.nol_b, s.nop)
    filename = ('[{}]{}.csv'.format(s.state_num, eigenvalues[s.state_num]))
    write_file(path_ti + 'RhoStates(PosBasis)/', filename, rho_pbasis)

    rho_ebasis = transformation(rho_pbasis, h_bd_evecs)
    filename = ('[{}]{}.csv'.format(s.state_num, eigenvalues[s.state_num]))
    write_file(path_ti + 'RhoStates(EnBasis)/', filename, rho_ebasis)

    tot_time2 = time()


    status(7, tot_time2 - tot_time1)
    return True


if __name__ == '__main__':
    """
    init_values = [
    Total no. of particles(nop),
    Dimension of lattice(ndims),
    t_initial,
    t_final,
    t_steps,
    initial eigenvector no.,
    """

    init_values = [1, 6, 0.0, 10.0, 50, 0]

    # Define lattices A and B
    lat_a = np.genfromtxt('a.txt', dtype=int)
    lat_b = np.genfromtxt('b.txt', dtype=int)

    main_states(init_values, lattice_a=lat_a, lattice_b=lat_b)
