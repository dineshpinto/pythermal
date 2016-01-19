# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Time evolving hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import print_function, division

import math as mt
import os
import time

import numpy as np
import scipy.linalg as la

import output
import subroutine1
import subroutine2
import subroutine3

__author__ = "Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics"


class System:
    def __init__(self, val, lat_a=None, lat_b=None):
        """
        Class stores metadata about the system
        :param val: List containing initial state values of the system
        :param lat_a: Lattice sites in A (only used when manually defining lattice)
        :param lat_b: Lattice sites in B (only used when manually defining lattice)

        """
        # No of particles
        self.nop = int(val[0])

        # Shape of square 2D array i.e. nsa = 2(2x2), 3(3x3)
        self.nsa = int(val[1])

        # Time Evolution - starting time, ending time and no. of time steps
        self.t_initial = float(val[3])
        self.t_final = float(val[4])
        self.t_steps = int(val[5])

        # Show images during execution (0=NO & 1=YES)
        self.checkbox = val[7]

        # Set initial state as an eigenvector of entire system (0=NO & 1=YES)
        self.checkbox2 = val[8]

        # Lattice sites to delete for particular values of nsa & nol_a
        self.lat_del_pos, self.lat_del_pos_a = self.lattice_generator

        # Automatically define lattice sites
        if lat_a is None or lat_b is None:
            # No. of sites in sub-lattice A
            self.nol_a = int(val[2])

            # No of  lattice sites before deletion eg. nsa = 3 => nol = 9
            self.nol = self.nsa * self.nsa

            # No. of sites in sub-lattice B
            self.nol_b = self.nol - (self.nol_a + len(self.lat_del_pos)) + 1

            # Site joining sub-lattice A and B (numbered after deleting sites)
            self.link_pos = mt.sqrt(self.nol_a) * self.nsa - (self.nsa - mt.sqrt(self.nol_a))

        # Manually define lattice sites
        else:
            # No. of sites in sub-lattice A
            self.nol_a = len(lat_a)
            # No of  lattice sites before deletion eg. nsa = 3 => nol = 9
            self.nol = len(lat_a) + len(lat_b)

            # No. of sites in sub-lattice B
            self.nol_b = len(lat_b)

        # Lattice before deleting sites
        self.lat = np.arange(1, self.nol + 1, dtype=np.int32)

    @property
    def timesteps(self):
        """
        :return: Array of times the system will be sampled

        """
        # Time gap between successive time steps
        delta_t = (self.t_final - self.t_initial) / self.t_steps

        # Array storing various times
        timestep_array = np.arange(self.t_initial, self.t_final, delta_t)

        return timestep_array

    @property
    def lattice_generator(self):
        """
        Generates lattice for defined nsa and nol_a. Assumes square blocks for A and B. [DOES NOT BREAK SYMMETRY]
        :return: Indices of lattice sites to be deleted to form whole lattice
        :return: Indices of lattice sites to be deleted to form A

        """
        if self.nsa == 3 and self.nol_a == 4:
            self.lat_del_pos = np.array([3, 7])
            self.lat_del_pos_a = np.array([3, 6, 7, 8, 9])

        elif self.nsa == 4 and self.nol_a == 4:
            self.lat_del_pos = np.array([3, 4, 9, 13])
            self.lat_del_pos_a = np.array([3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

        elif self.nsa == 4 and self.nol_a == 9:
            self.lat_del_pos = np.array([4, 8, 13, 14])
            self.lat_del_pos_a = np.array([4, 8, 12, 13, 14, 15, 16])

        elif self.nsa == 5 and self.nol_a == 4:
            self.lat_del_pos = np.array([3, 4, 5, 11, 16, 21])
            self.lat_del_pos_a = np.array([3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                           25])

        elif self.nsa == 5 and self.nol_a == 9:
            self.lat_del_pos = np.array([4, 5, 9, 10, 16, 17, 21, 22])
            self.lat_del_pos_a = np.array([4, 5, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])

        elif self.nsa == 5 and self.nol_a == 16:
            self.lat_del_pos = np.array([5, 10, 15, 21, 22, 23])
            self.lat_del_pos_a = np.array([5, 10, 15, 21, 21, 22, 23, 24, 25])

        else:
            raise Exception('Lattice shape {0}x{0} with {1} latice sites in A not supported. Unable to delete sites.'.
                            format(self.nsa, self.nol_a))

        return self.lat_del_pos, self.lat_del_pos_a

    @property
    def folder_path_ti(self):
        """
        :return: Path for storing time independent variables
        """
        return 'Output_PyThermal/TI-{}_{}_{}/'.format(self.nop, self.nsa, self.nol_a)

    @property
    def folder_path_td(self):
        """
        :return: Path for storing time dependent variables
        """
        return 'Output_PyThermal/TD-{}_{}_{}_{}/'.format(self.nop, self.nsa, self.nol_a, self.t_final)

    @staticmethod
    def plotting_metadata():
        """
        Stores metadata for matplotlib plots
        :return: Filename of images
        :return: Image titles
        :return: y axis labels
        :return: x axis labels
        :return: y axis limits

        """
        fnames = ['VN_Entropy_B', 'Trace2_B', 'Avg_A', 'Avg_B', 'Avg_AB']

        titles = [r'Von-Neumann entropy ($S_{VN}$) vs time ($\tau$)', r'Purity ($tr(\rho^2))$) vs time ($\tau$)',
                  r'Avg. particles in A vs time ($\tau$)', r'Avg. particles in B vs time ($\tau$)',
                  r'Avg. particles in A and B vs time ($\tau$)']

        y_labels = [r'Von-Neumann Entropy $[S_{VN} = - tr(\rho \ln(\rho))] \rightarrow$',
                    r'Purity $[tr(\rho^2))] \rightarrow$', r'Avg. particles in A', r'Avg. particles in B',
                    r'Avg. particles in A and B']

        x_labels = [r'Time $[\tau]\rightarrow$']

        y_limits = [(0.0, 3.0), (-1.0, 2.0), (0.0, 5.0)]

        return fnames, titles, y_labels, x_labels, y_limits

    @staticmethod
    def variable_names():
        """
        :return: Names of variables for storing on hard disk

        """
        names = ['Hamiltonian.csv', 'Hamiltonian_A.csv', 'Eigenvalues.csv',
                 'Eigenvectors.csv', 'Eigenvalues_A.csv', 'Eigenvectors_A.csv', 'Psi.csv', 'Sum_A.csv', 'Sum_B.csv',
                 'VN_Entropy_B.csv', 'VN_Trace2_B']

        return names

    @property
    def check_existence(self):
        """
        Checks whether variables exists on hard disk
        :return: Names of variables for storing on hard disk
        :return: Integer list whether files exists on hard disk(1) or not(0)

        """
        names = self.variable_names()
        existence = [0] * len(names)

        for idx, name in enumerate(names):

            if os.path.isfile(self.folder_path_ti + name):
                existence[idx] = 1
                print('{} exists'.format(name))

            elif os.path.isfile(self.folder_path_td + name):
                existence[idx] = 1
                print('{} exists'.format(name))

            else:
                print('{} does not exist'.format(name))

        print('\n')
        return names, existence


def check_lattice(initial_values):
    """
    Runs checks to make sure all inputs are valid. Raises ValueError if not.
    :param initial_values: initial values from main()

    """
    if initial_values[0] <= 0:
        raise ValueError('No. of particles should be greater than 0')

    if initial_values[0] > initial_values[2] and not initial_values[8]:
        raise ValueError('Too many particles [{}] for sub-lattice A [{}]'.format(initial_values[0], initial_values[2]))

    if initial_values[0] > initial_values[1] ** 2:
        raise ValueError('Too many particles [{}] for lattice [{}]'.format(initial_values[0], initial_values[1] ** 2))

    if initial_values[1] < 3:
        raise ValueError('Shape of lattice must be at least 3')

    if initial_values[4] <= initial_values[3]:
        raise ValueError(
                'Final time [{}] must be greater than initial time [{}]'.format(initial_values[4], initial_values[3]))

    if initial_values[5] <= 0:
        raise ValueError('No. of time steps has to be greater than 0')


def main(initial_values, lattice_a=None, lattice_b=None):
    """
    Contains calls to/control of all functions in program.
    :param lattice_a: Array of lattice sites in A
    :param lattice_b: Array of lattice sites in B
    :param initial_values:
    :return: True if execution was successful

    """
    check_lattice(initial_values)

    if initial_values[9]:
        s = System(initial_values, lattice_a, lattice_b)
    else:
        s = System(initial_values)

    path_ti, path_td = s.folder_path_ti, s.folder_path_td
    image_name, titles, y_labels, x_labels, y_limits = s.plotting_metadata()

    names, existence = s.check_existence

    # -----Sub-Routine 1 (Eigenstates, Hamiltonian, Eigenvalues and Eigenvectors)-----

    # Eigenstates
    if initial_values[9]:
        eigenstates, nos = subroutine1.eigenstates_lattice(s.lat, s.nop)
        eigenstates_a, nos_a = subroutine1.eigenstates_lattice(lattice_a, s.nop)
    else:
        eigenstates, nos = subroutine1.eigenstates_lattice(s.lat, s.nop, s.lat_del_pos)
        eigenstates_a, nos_a = subroutine1.eigenstates_lattice(s.lat, s.nop, s.lat_del_pos_a)

    output.status(1)

    output.write_file(path_ti, 'Eigenstates.csv', eigenstates, fmt='%1d')
    output.write_file(path_ti, 'Eigenstates_A.csv', eigenstates_a, fmt='%1d')

    # Hamiltonian
    h_time1 = time.time()

    if existence[0]:
        hamiltonian = output.read_file(path_ti, names[0])
    else:
        print('Hamiltonian...')
        hamiltonian = subroutine1.parallel_call_hamiltonian(eigenstates, nos, s.nsa, s.nop)
        output.write_file(path_ti, names[0], hamiltonian, fmt='%1d')

    if existence[1]:
        hamiltonian_a = output.read_file(path_ti, names[1])
    else:
        hamiltonian_a = subroutine1.parallel_call_hamiltonian(eigenstates_a, nos_a, s.nsa, s.nop)
        output.write_file(path_ti, names[1], hamiltonian_a, fmt='%1d')

    h_time2 = time.time()
    output.status(2, h_time2 - h_time1)

    # Eigenvalues and Eigenvectors
    e_time1 = time.time()

    if existence[2] and existence[3]:
        # eigenvalues = Output.read_file(path_ti, names[2])
        eigenvectors = output.read_file(path_ti, names[3], dtype=complex)

    else:
        print('Diagonalizing...')
        eigenvalues, eigenvectors = subroutine1.eigenvalvec(hamiltonian)
        output.write_file(path_ti, names[2], eigenvalues)
        output.write_file(path_ti, names[3], eigenvectors)

    # Eigenvalues and Eigenvectors of A
    if existence[4] and existence[5]:
        # eigenvalues_a = Output.read_file(path_ti, names[4])
        eigenvectors_a = output.read_file(path_ti, names[5], dtype=complex)

    else:
        print('Diagonalizing A...')
        eigenvalues_a, eigenvectors_a = subroutine1.eigenvalvec(hamiltonian_a)
        output.write_file(path_ti, names[4], eigenvalues_a)
        output.write_file(path_ti, names[5], eigenvectors_a)

    e_time2 = time.time()
    output.status(3, e_time2 - e_time1)

    # ------Sub-Routine 2 (State Relabelling and Initial State)-----

    # State Relabelling
    r_time1 = time.time()
    if initial_values[9]:
        re_states = subroutine2.relabel(eigenstates, s.nop, s.nol_b, link_pos=None, lat_b=lattice_b)
    else:
        re_states = subroutine2.relabel(eigenstates, s.nop, s.nol_b, link_pos=s.link_pos, lat_b=None)
    r_time2 = time.time()
    output.status(4, r_time2 - r_time1)
    output.write_file(path_ti, 'RelabelledStates.csv', re_states)

    # Initial State
    state_num = int(initial_values[6])

    if s.checkbox2:
        psi_initial = eigenvectors[:, state_num] / la.norm(eigenvectors[:, state_num])
    else:
        psi_initial = subroutine2.init_state(eigenvectors_a, re_states, nos, s.nop, state_num)

    # -----Sub-Routine 3 (Time Evolution and Von-Neumann Entropy)-----

    # Time Evolution
    evo_time1 = time.time()

    if existence[6] and existence[7] and existence[8]:
        psi_t = output.read_file(path_td, names[6], dtype=complex)
        sum_a = output.read_file(path_td, names[7])
        sum_b = output.read_file(path_td, names[8])

    else:
        print('Time evolution...')
        psi_t, sum_a, sum_b = subroutine3.time_evolution(psi_initial, hamiltonian, nos, s.timesteps, re_states,
                                                         s.nop)
        output.write_file(path_td, names[6], psi_t)
        output.write_file(path_td, names[7], sum_a)
        output.write_file(path_td, names[8], sum_b)

    evo_time2 = time.time()
    output.status(5, evo_time2 - evo_time1)

    # Von-Neumann Entropy
    vn_time1 = time.time()

    if existence[9] and existence[10]:
        vn_entropy_b = output.read_file(path_td, names[9], dtype=complex)
        vn_trace2_b = output.read_file(path_td, names[10])
    else:
        print('Von-Neumann Entropy...')
        vn_entropy_b, vn_trace2_b = subroutine3.von_neumann_b(psi_t, re_states, nos, s.nol_b, s.nop)
        output.write_file(path_td, names[9], vn_entropy_b)
        output.write_file(path_td, names[10], vn_trace2_b)

    vn_time2 = time.time()
    output.status(6, vn_time2 - vn_time1)

    # -----Output-----
    x = s.timesteps

    output.plot(x, vn_entropy_b, titles[0], y_labels[0], x_labels[0], y_limits[0], path_td, image_name[0], s.checkbox)
    output.plot(x, vn_trace2_b, titles[1], y_labels[1], x_labels[0], y_limits[0], path_td, image_name[1], s.checkbox)
    output.plot(x, sum_a, titles[2], y_labels[2], x_labels[0], y_limits[2], path_td, image_name[2], s.checkbox)
    output.plot(x, sum_b, titles[3], y_labels[3], x_labels[0], y_limits[2], path_td, image_name[3], s.checkbox)
    output.plot(x, sum_b + sum_a, titles[4], y_labels[4], x_labels[0], y_limits[2], path_td, image_name[4], s.checkbox)

    # -----Terminate-----
    output.status(7)
    return True


if __name__ == '__main__':
    # [nop, nsa, nol_a, t_initial, t_final, t_steps, Show images(1=YES), initial psi(1=eigenstate of entire system),
    # eigenvector no., manually define lattice(1=YES), link position]
    init_values = [1, 4, 4, 0.0, 50.0, 100, 0, 0, 0, 0, 0]
    l_a = np.array([])
    l_b = np.array([])
    main(init_values)
