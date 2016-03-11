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

if cpu_count() > 4:
    if 'OPENBLAS_NUM_THREADS' not in os.environ:
        os.environ['OPENBLAS_NUM_THREADS'] = '16'

import numpy as np
from scipy.linalg import norm

from output import status, write_file, read_file
from routines import (position_states, hamiltonian_parallel, diagonalize,
                      relabel, rho_b_pbasis, transformation, naive_thermal,
                      h_block_diagonal)


__author__ = "Thermalization and Quantum Entanglement Project Group, SSCTP"
__version__ = "v1.9.0-states"


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
        :return: Path for storing output
        """
        return ('../pythermal_output/{}_{}_{}_{}/'
                .format(self.nop, self.n_dim, self.nol_a, self.nol_b))

    def check_existence(self, names):
        """
        Checks whether variables exists on hard disk.
        Could be deprecated for try-except statements.
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

    def check_system(self):
        """
        Runs checks to make sure all inputs are valid.
        Raises ValueError if not.
        """
        if not self.nop > 0:
            raise ValueError('No. of particles should be greater than 0')

        if not self.nop < self.n_dim ** 2:
            raise ValueError('Too many particles [{}] for lattice [{}]'
                             .format(self.nop, self.n_dim ** 2))
    
        if not self.n_dim > 3:
            raise ValueError('Shape of lattice must be at least 3')


def main_states(initial_values, chosen_eigenstates, lattice_a, lattice_b):
    """
    Contains calls to/control of all functions in program.
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
    pos_states_ab, nos_ab = position_states(s.lattice, s.nop)
    pos_states_a, nos_a = position_states(lattice_a, s.nop)
    write_file(path, 'PositionStates_AB.csv', pos_states_ab, fmt='%1d')
    write_file(path, 'PositionStates_A.csv', pos_states_a, fmt='%1d')

    # Eigenvalues and Eigenvectors
    e_time1 = time()
    try:
        eigenvalues = read_file(path, 'Eigenvalues_AB.csv')
        eigenvectors = read_file(path, 'Eigenvectors_AB.csv', dtype=complex)
    except IOError:
        try:
            hamiltonian = read_file(path, 'Hamiltonian_AB.csv', dtype=int)
        except IOError:
            print("Hamiltonian...")
            hamiltonian = hamiltonian_parallel(s.lattice, s.n_dim, s.nop)
            write_file(path, 'Hamiltonian_AB.csv', hamiltonian, fmt='%1d')
            print("Hamiltonian done!")
        print("Diagonalizing...")
        eigenvalues, eigenvectors = diagonalize(hamiltonian)
        write_file(path, 'Eigenvalues_AB.csv', eigenvalues)
        write_file(path, 'Eigenvectors_AB.csv', eigenvectors)
        print("Diagonalizing done!")
    e_time2 = time()
    status(e_time2 - e_time1)

    # Block diagonal
    bd_time1 = time()
    try:
        h_bd_evecs = read_file(path, 'Hamiltonian_B_BD_Evecs.csv',
                               dtype=complex)
    except IOError:
        try:
            h_bd = read_file(path, 'Hamiltonian_B_BD.csv')
        except IOError:
            print("Block diagonal...")
            h_bd = h_block_diagonal(lattice_b, s.n_dim, s.nop)
            write_file(path, 'Hamiltonian_B_BD.csv', h_bd, fmt='%1d')
            print("Block diagonal done!")
        print("Diagonalizing block diagonal...")
        _, h_bd_evecs = diagonalize(h_bd)
        write_file(path, 'Hamiltonian_B_BD_Evecs.csv', h_bd_evecs)
    bd_time2 = time()
    status(bd_time2 - bd_time1)

    # State Relabelling
    print("Relabelling...")
    labels = relabel(pos_states_ab, s.nop, s.nol_b, lat_a=lattice_a)
    write_file(path, 'RelabelledStates.csv', labels)

    for state_num in chosen_eigenstates:
        print("Eigenstate = {}/{}".format(state_num, nos_ab))
        state = eigenvectors[state_num] / norm(eigenvectors[state_num])

        # Density Matrix
        dm_time1 = time()
        rho_fname = '[{}]{}.csv'.format(state_num, eigenvalues[state_num])
        try:
            rho_ebasis = read_file(path + 'RhoStates(EnergyBasis)/',
                                   rho_fname, dtype=complex)
        except IOError:
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
            print("Transforming DM to Energy basis...")
            rho_ebasis = transformation(rho_pbasis, h_bd_evecs)
            write_file(path + 'RhoStates(EnergyBasis)/', rho_fname, rho_ebasis)
        dm_time2 = time()
        status(dm_time2 - dm_time1)

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


if __name__ == '__main__':
    """
    init_values = [Total no. of particles(nop), Dimension of lattice(ndims)]
    chosen_estate = [Eigenstates  for which to compute DM's]
    """
    init_values = [2, 6]
    chosen_e_states = [_ for _ in range(10, 100, 2)]

    main_states(init_values, chosen_e_states,
                lattice_a=np.genfromtxt('a.txt', dtype=int),
                lattice_b=np.genfromtxt('b.txt', dtype=int))
