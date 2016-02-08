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
import time

if 'OPENBLAS_MAIN_FREE' not in os.environ:
    os.environ['OPENBLAS_MAIN_FREE'] = '1'

if 'MKL_NUM_THREADS' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = '1'

if 'OPENBLAS_NUM_THREADS' not in os.environ:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.linalg as la

from output import status, write_file, read_file, plotting
from routines import (position_states, parallel_call_h, eig, relabel,
                      initial_state, vn_entropy_b, time_evolution,
                      avg_particles, h_block_diagonal)


__author__ = "Thermalization and Quantum Entanglement Project Group, SSCTP"
__version__ = "v1.5.0"


class MetaSystem:
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
        self.ndims = int(initial_value[1])

        # Time Evolution - starting time, ending time and no. of time steps
        self.t_initial = float(initial_value[2])
        self.t_final = float(initial_value[3])
        self.t_steps = int(initial_value[4])

        # Initial state for psi_initial
        self.state_num = int(initial_value[5])

        # Set initial state as an eigenvector of entire system (0=NO & 1=YES)
        self.eigenvector_of_ab = int(initial_value[6])

        # No. of sites in sub-lattice A
        self.nol_a = len(lattice_a)

        # No. of sites in sub-lattice B
        self.nol_b = len(lattice_b)

        # No of  lattice sites before deletion eg. nsa = 3 => nol = 9
        self.nol = len(lattice_a) + len(lattice_b)

        # Lattice before deleting sites
        self.lat = np.arange(1, self.nol + 1, dtype=np.int32)

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
    def folder_path_ti(self):
        """
        :return: Path for storing time independent variables
        """
        return ('../Output_PyThermal/TI-{}_{}_{}/'
                .format(self.nop, self.ndims, self.nol_a))

    @property
    def folder_path_td(self):
        """
        :return: Path for storing time dependent variables
        """
        return ('../Output_PyThermal/TD-{}_{}_{}_{}/'
                .format(self.nop, self.ndims, self.nol_a, self.t_final))

    @staticmethod
    def variable_names():
        """
        :return: Names of variables for storing on hard disk

        """
        names = ['Hamiltonian.csv', 'Hamiltonian_A.csv', 'Eigenvalues.csv',
                 'Eigenvectors.csv', 'Eigenvalues_A.csv', 'Eigenvectors_A.csv',
                 'Psi_t.csv', 'Avg_A.csv', 'Avg_B.csv', 'VN_Entropy_B.csv',
                 'VN_Trace2_B']

        return names

    @property
    def check_existence(self):
        """
        Checks whether variables exists on hard disk.
        :return: Integer list whether files exists on hard disk(1) or not(0)

        """
        names = self.variable_names()
        existence = [False] * len(names)

        for idx, name in enumerate(names):

            if os.path.isfile(self.folder_path_ti + name):
                existence[idx] = True
                print('{} exists at {}'.format(name, self.folder_path_ti))

            elif os.path.isfile(self.folder_path_td + name):
                existence[idx] = True
                print('{} exists at {}'.format(name, self.folder_path_td))

            else:
                print('{} does not exist'.format(name))
        print('')

        return existence


# :TODO: Rewrite check_lattice for manual definition of A and B
def check_lattice(initial_values, options):
    """
    Runs checks to make sure all inputs are valid. Raises ValueError if not.
    :param options: List of options passed for execution
    :param initial_values: List of initial values for system

    """
    if not initial_values[0] > 0:
        raise ValueError('No. of particles should be greater than 0')

    if not initial_values[0] < initial_values[2] and not options[1]:
        raise ValueError('Too many particles [{}] for sub-lattice A [{}]'
                         .format(initial_values[0], initial_values[2]))

    if not initial_values[0] < initial_values[1] ** 2:
        raise ValueError('Too many particles [{}] for lattice [{}]'
                         .format(initial_values[0], initial_values[1] ** 2))

    if not initial_values[1] > 3:
        raise ValueError('Shape of lattice must be at least 3')

    if not initial_values[4] >= initial_values[3]:
        raise ValueError('Final time [{}] must be more than initial time [{}]'
                         .format(initial_values[4], initial_values[3]))

    if not initial_values[5] > 0:
        raise ValueError('No. of time steps has to be greater than 0')


def main(initial_values, options, lattice_a=None, lattice_b=None):
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
    if lattice_a is None or lattice_b is None:
        raise ValueError('Enter both lattice A and lattice B')

    s = MetaSystem(initial_values, lattice_a, lattice_b)
    path_ti, path_td = s.folder_path_ti, s.folder_path_td
    names = s.variable_names()

    if options[1]:
        existence = s.check_existence
    else:
        existence = [False] * len(names)

    tot_time1 = time.time()

    # Eigenstates
    pos_states, nos = position_states(s.lat, s.nop)
    pos_states_a, nos_a = position_states(lattice_a, s.nop)
    status(1)
    write_file(path_ti, 'PositionStates.csv', pos_states, fmt='%1d')
    write_file(path_ti, 'PositionStates_A.csv', pos_states_a, fmt='%1d')

    # Hamiltonian
    h_time1 = time.time()
    if existence[0]:
        # Reads Hamiltonian from hard disk
        hamiltonian = read_file(path_ti, names[0])
    else:
        # Otherwise generates Hamiltonian
        print('Hamiltonian...')
        hamiltonian = parallel_call_h(pos_states, nos, s.ndims, s.nop)
        write_file(path_ti, names[0], hamiltonian, fmt='%1d')

    if existence[1]:
        hamiltonian_a = read_file(path_ti, names[1])
    else:
        hamiltonian_a = parallel_call_h(pos_states_a, nos_a, s.ndims, s.nop)
        write_file(path_ti, names[1], hamiltonian_a, fmt='%1d')
    h_time2 = time.time()
    status(2, h_time2 - h_time1)

    # Block diagonal of Hamiltonian
    h_bd = h_block_diagonal(lattice_b, s.ndims, s.nop)
    write_file(path_ti, 'H_BD.csv', h_bd, fmt='%1d')

    # Eigenvalues and Eigenvectors
    e_time1 = time.time()
    if existence[2] and existence[3]:
        eigenvalues = read_file(path_ti, names[2])
        eigenvectors = read_file(path_ti, names[3], dtype=complex)
    else:
        print('Diagonalizing...')
        eigenvalues, eigenvectors = eig(hamiltonian)
        write_file(path_ti, names[2], eigenvalues)
        write_file(path_ti, names[3], eigenvectors)

    # Eigenvalues and Eigenvectors of A
    if existence[4] and existence[5]:
        # eigenvalues_a = output.read_file(path_ti, names[4])
        eigenvectors_a = read_file(path_ti, names[5], dtype=complex)
    else:
        print('Diagonalizing A...')
        eigenvalues_a, eigenvectors_a = eig(hamiltonian_a)
        write_file(path_ti, names[4], eigenvalues_a)
        write_file(path_ti, names[5], eigenvectors_a)
    e_time2 = time.time()
    status(3, e_time2 - e_time1)

    # State Relabelling
    r_time1 = time.time()
    labels = relabel(pos_states, s.nop, s.nol_b, lat_a=lattice_a)
    r_time2 = time.time()
    status(4, r_time2 - r_time1)
    write_file(path_ti, 'RelabelledStates.csv', labels)

    # :TODO: Finds density matrix for a given number of pos_states
    # states, eigenvals = state_initializer(eigenvalues, eigenvectors, num=66)
    # write_file(path_ti, 'states.csv', states)
    # for idx, state in enumerate(states):
    #     d_matrix = density_matrix_b(labels, state, nos, s.nol_b, s.nop)
    #     filename = 'density_b[{}].csv'.format(eigenvals[idx])
    #     write_file(path_ti, filename, d_matrix)

    # Initial State
    if s.eigenvector_of_ab:
        # Initial state as an eigenvector of the entire system
        chosen_eigenvector = eigenvectors[:, s.state_num]
        psi_initial = chosen_eigenvector / la.norm(chosen_eigenvector)
    else:
        # Initial state as eigenvector of A
        psi_initial = initial_state(eigenvectors_a, labels, nos, s.nop,
                                    s.state_num)

    # Time Evolution
    evo_time1 = time.time()
    if existence[6] and existence[7] and existence[8]:
        psi_t = read_file(path_td, names[6], dtype=complex)
    else:
        print('Time evolving...')
        psi_t = time_evolution(psi_initial, hamiltonian, nos, s.timesteps)
        write_file(path_td, names[6], psi_t)
    evo_time2 = time.time()
    status(5, evo_time2 - evo_time1)

    # Average number of particles in A and B
    if existence[7] and existence[8]:
        avg_a = read_file(path_td, names[7])
        avg_b = read_file(path_td, names[8])
    else:
        avg_a, avg_b = avg_particles(psi_t, s.timesteps, labels, s.nop)
        write_file(path_td, names[7], avg_a)
        write_file(path_td, names[8], avg_b)

    # Von-Neumann Entropy
    vn_time1 = time.time()
    if existence[9] and existence[10]:
        entropy_b = read_file(path_td, names[9], dtype=complex)
        tr_sqr_b = read_file(path_td, names[10])
    else:
        print('Von-Neumann Entropy...')
        entropy_b, tr_sqr_b = vn_entropy_b(psi_t, labels, nos, s.nol_b, s.nop)
        write_file(path_td, names[9], entropy_b)
        write_file(path_td, names[10], tr_sqr_b)
    vn_time2 = time.time()
    status(6, vn_time2 - vn_time1)

    plotting(entropy_b, tr_sqr_b, avg_a, avg_b, path_td, s.timesteps,
             options[0])

    tot_time2 = time.time()
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
    initial psi(1=eigenstate of entire system)]

    opts = [
    Show images(1=YES),
    Check for files on disk(1=YES)]
    
    """
    init_values = [2, 5, 0.0, 10.0, 50, 0, 0]
    opts = [0, 0]

    # Define lattices A and B
    lat_a = np.genfromtxt('a.txt', dtype=int)
    lat_b = np.genfromtxt('b.txt', dtype=int)
    main(init_values, opts, lat_a, lat_b)
